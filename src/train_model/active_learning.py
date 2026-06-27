import copy
import json
import os
from pathlib import Path
from typing import List, Callable, Optional, Any, Union

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.dataset_utils.datasets import PartialSequenceDataset
from src.train_model.acquisition_strategies.acquisition_strategy import AcquisitionStrategy


class ActiveLearner:
    # --- Changes to __init__ ---
    def __init__(self,
                 model_class: Callable[[], nn.Module],
                 loss_fn: Callable[[Any, Any], torch.Tensor],
                 dataset: PartialSequenceDataset,
                 seed_indices: Union[List[int], np.ndarray],
                 val_indices: Union[List[int], np.ndarray],
                 pool_indices: Union[List[int], np.ndarray],
                 acquisition: AcquisitionStrategy,
                 device: Optional[torch.device] = None,
                 retrain_from_scratch: bool = True,
                 ensemble_size: int = 1,
                 acquire_full_time: bool = True,
                 verbose: bool = True):
        """
        dataset: PartialSequenceDataset (must expose initialize_seed and observe_next_step).
        seed_indices: initially fully labeled (obs_lens = T).
        pool_indices: indices eligible for stepwise revealing (excluding seeds).
        """
        self.model_class = model_class
        self.loss_fn = loss_fn
        self.dataset = dataset
        self.max_time = dataset.max_t
        self.acquire_full_time = acquire_full_time
        # set up labeled/pool/val indices
        if isinstance(seed_indices, np.ndarray):
            self.labeled_indices = seed_indices.tolist()
        else:
            self.labeled_indices = seed_indices
        if isinstance(pool_indices, np.ndarray):
            self.pool_indices = pool_indices.tolist()
        else:
            self.pool_indices = pool_indices
        if isinstance(val_indices, np.ndarray):
            self.val_indices = val_indices.tolist()
        else:
            self.val_indices = val_indices
        self.acquisition = acquisition
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.retrain_from_scratch = retrain_from_scratch
        self.ensemble_size = ensemble_size
        self.verbose = verbose

        # ensure dataset seed initialization if available
        if hasattr(self.dataset, 'initialize_seed'):
            self.dataset.initialize_seed(self.labeled_indices)

        # bookkeeping counters
        # total acquisitions done so far (cumulative)
        self.total_acquisitions = 0

        # storage for metrics - extended to include events & observed timesteps per round
        self.history = {
            'rounds': [],  # keeps round_info dicts
            'training_losses': [],  # list per round (list of epoch losses)
            'validation_losses': [],  # list per round
            'acquisition_sizes': [],  # number acquired per round
            'events_observed': [],  # total number of observed "events" at end of each round
            'observed_timesteps': []  # per-round mapping idx -> observed timesteps (list of dicts)
        }

    # --- New helper: count observed events (used when storing round info) ---
    def _count_events_observed(self):
        """
        Counts number of events (y == 1) among observed timesteps.
        Handles dataset.y being numpy or torch tensor.
        Only counts up to dataset.obs_lens[idx] for each idx.
        Returns integer total.
        """
        total_events = 0
        N = len(self.dataset)
        # try to get y as numpy for speed, otherwise handle per-sample
        y_all = getattr(self.dataset, 'y', None)
        for idx in range(N):
            obs_len = int(self.dataset.obs_lens[idx])
            if obs_len <= 0:
                continue
            if y_all is None:
                # fallback: fetch item via dataset[idx] (may be expensive)
                try:
                    item = self.dataset[idx]
                    # assume dataset[idx] returns (x_prefix, label_obj, idx) or similar;
                    # we cannot reliably reconstruct full y here, so skip counting
                    continue
                except Exception:
                    continue
            else:
                # y_all exists; handle torch or numpy
                yi = y_all[idx]
                if isinstance(yi, torch.Tensor):
                    # slice [:obs_len] and sum equality to 1
                    total_events += int((yi[:obs_len] == 1).sum().item())
                else:
                    # assume numpy-like
                    total_events += int((np.array(yi[:obs_len]) == 1).sum())
        return total_events

    def _make_validation_dataloader(self, validation_indices: List[int], batch_size: int, shuffle: bool = True):
        subset = Subset(self.dataset, validation_indices)
        return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    def _make_train_dataloader(self, validation_indices: List[int], batch_size: int, shuffle: bool = True):
        indices = list(set((self.dataset.obs_lens > 1).nonzero().squeeze().tolist()) - set(validation_indices))
        subset = Subset(self.dataset, indices)
        return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    def _train_one_epoch(self, model, lr, weight_decay, train_loader, train_indices):
        model.train()
        total_loss = 0.0
        count = 0

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        for batch in train_loader:
            # now batch is (x_prefix, label_obj, idx)
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                x_prefix, label_objs, idxs = batch
            else:
                raise RuntimeError("Dataset must return (x_prefix, label_obj, idx).")
            x_prefix = x_prefix.clone().to(self.device)
            max_time = x_prefix.shape[1]
            missing_indicator = (torch.arange(max_time, device=x_prefix.device) >= label_objs['obs_len'][:, None].to(x_prefix.device))
            x_prefix[missing_indicator] = 0
            optimizer.zero_grad()
            out = model(x_prefix)  # model should accept variable-length prefixes (pack/pad handled externally)

            loss = self.loss_fn(out, (x_prefix, label_objs, idxs))  # loss adapter must accept label_objs
            loss.backward()
            # --- ADD THIS LINE ---
            # This scales down the gradient vector if its norm exceeds 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # ---------------------
            optimizer.step()
            total_loss += float(loss.detach().cpu().item()) * x_prefix.shape[0]
            count += x_prefix.shape[0]

        train_loss = total_loss / max(1, count)

        return train_loss

    def _eval(self, model, indices, batch_size=64):
        model.eval()
        loader = self._make_validation_dataloader(indices, batch_size, shuffle=False)
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    x_prefix, label_objs, idxs = batch
                else:
                    raise RuntimeError("Dataset must return (x_prefix, label_obj, idx).")
                x_prefix = x_prefix.to(self.device)
                out = model(x_prefix)
                loss = self.loss_fn(out, (x_prefix, label_objs, idxs))
                total_loss += float(loss.detach().cpu().item()) * x_prefix.shape[0]
                count += x_prefix.shape[0]
        return total_loss / max(1, count)

    def _fit_model(self, model: nn.Module, labeled_indices: List[int], val_indices: List[int],
                   lr: float = 1e-4, max_epochs: int = 100, patience: int = 10,
                   batch_size: int = 64, eval_batch_size: int = 2048, weight_decay: float = 0.0,
                   checkpoint_path: Optional[Path] = None, do_tqdm=True):
        model = model.to(self.device)

        train_loader = self._make_train_dataloader(val_indices, batch_size=batch_size, shuffle=True)

        best_val = float('inf')
        best_state = None
        patience_counter = 0
        train_losses = []
        val_losses = []
        iterator = tqdm(range(1, max_epochs + 1), desc="Training") if do_tqdm else range(1, max_epochs + 1)

        for epoch in iterator:

            train_loss = self._train_one_epoch(model, lr, weight_decay, train_loader, labeled_indices)

            val_loss = self._eval(model, val_indices, batch_size=eval_batch_size)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if self.verbose:
                print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

            elif do_tqdm:
                iterator.set_postfix({
                    'train_loss': f'{train_loss:.4f}',
                    'val_loss': f'{val_loss:.4f}'
                })

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                if self.verbose:
                    print(f"Early stopping at epoch {epoch} (patience {patience})")
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        if checkpoint_path is not None:
            torch.save({'model_state': model.state_dict(), 'best_val': best_val}, checkpoint_path)

        return model, train_losses, val_losses

    def acquire(self, model: nn.Module, k: int, batch_size: int = 64) -> List[int]:
        if len(self.pool_indices) == 0:
            return []
        k = min(k, len(self.pool_indices))
        selected = self.acquisition.select(model, self.dataset, self.pool_indices, k=k, batch_size=batch_size)
        return selected

    # --- Changes to store_state ---
    def store_state(self, model, outdir, round):
        """
        Save AL state including:
         - labeled / pool indices
         - history
         - current round
         - dataset.obs_lens (so we can resume exact reveal state)
         - total_acquisitions
         - model weights
        """
        os.makedirs(outdir, exist_ok=True)
        # make sure obs_lens is serializable (list of ints)
        obs_lens = [int(x) for x in getattr(self.dataset, 'obs_lens')]
        state = {
            'labeled_indices': self.labeled_indices,
            'pool_indices': self.pool_indices,
            'history': self.history,
            'current_round': round,
            'obs_lens': obs_lens,
            'total_acquisitions': int(self.total_acquisitions),
        }
        torch.save({
            'state': state,
            'model_state': model.state_dict()
        }, Path(outdir) / 'al_state_latest.pt')

        # --- Changes to load_state ---

    def load_state(self, out_models_dir, update_steps=True):
        p = Path(out_models_dir) / 'al_state_latest.pt'
        if not p.exists():
            return None

        ckpt = torch.load(p, map_location=self.device)
        state = ckpt['state']

        self.labeled_indices = state['labeled_indices']
        self.pool_indices = state['pool_indices']
        self.history = state['history']
        last_round = state['current_round']

        # restore dataset observed lengths exactly so the stepwise reveal continues where it left off
        saved_obs = state.get('obs_lens', None)
        if saved_obs is not None and update_steps:
            for i in range(len(saved_obs)):
                for _ in range(saved_obs[i]):
                    self.dataset.observe_next_step([i])

        # restore cumulative acquisitions counter
        self.total_acquisitions = int(
            state.get('total_acquisitions', sum(self.history.get('acquisition_sizes', []))))

        model = self.model_class().to(self.device)
        model.load_state_dict(ckpt['model_state'])
        return model, last_round

    def compute_total_acquired(self, history_file_path: Union[Path, str] = 'al_history.json'):
        # Check if file exists first
        if not os.path.exists(history_file_path):
            print(f"Error: The file '{history_file_path}' was not found.")
            return

        try:
            with open(history_file_path, 'r') as f:
                history_data = json.load(f)

            # 1. Access the 'rounds' list based on: self.history['rounds']
            rounds = history_data.get('rounds', [])

            # 2. Sum the 'acquired_count' for each round
            # Based on: round_info['acquired_count'] = len(selected)
            total_acquired = sum(round_info.get('acquired_count', 0) for round_info in rounds)

            print(f"Successfully loaded {len(rounds)} rounds.")
            print(f"Total Acquired Count: {total_acquired}")

            return total_acquired

        except json.JSONDecodeError:
            print(f"Error: '{history_file_path}' is not a valid JSON file.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def __conduct_one_batch_one_time_acquire(self, selected_idxs: List[int]):
        # reveal one step per selected index
        event_flags = self.dataset.observe_next_step(selected_idxs)
        self.total_acquisitions += len(selected_idxs)

        # Add to labeled set if not present; remove from pool only when event observed or reached T
        for idx, ev in zip(selected_idxs, event_flags):
            if idx not in self.labeled_indices:
                self.labeled_indices.append(idx)
            # if event observed or obs_len == T -> remove from pool (no more revealing)
            if ev or (self.dataset.obs_lens[idx] >= self.dataset.T):
                if idx in self.pool_indices:
                    self.pool_indices.remove(idx)

    def __conduct_one_batch_full_time_acquire(self, selected_idxs: List[int], total_budget_left):
        assert (len(selected_idxs) - 1) * self.max_time < total_budget_left
        n_acquired = 0
        if len(selected_idxs) * self.max_time > total_budget_left:
            partial_select_idx = selected_idxs[-1]
            selected_idxs = selected_idxs[:-1]
            for _ in range(0, total_budget_left - (len(selected_idxs) - 1) * self.max_time):
                self.dataset.observe_next_step([partial_select_idx])
                n_acquired += 1
                # total_budget_left -= 1
                self.total_acquisitions += 1
        self.dataset.set_fully_observed(selected_idxs)
        for idx in selected_idxs:
            self.pool_indices.remove(idx)
        n_acquired += len(selected_idxs) * self.max_time
        # total_budget_left -= len(selected_idxs) * self.max_time
        self.total_acquisitions += len(selected_idxs) * self.max_time
        return n_acquired

    def __conduct_one_batch_acquire(self, model, total_budget_left, acquire_batch_size, samples_acquire_batch_size,
                                                                            n_repeated_samples_acquire):
        total_budget_left = copy.deepcopy(total_budget_left)
        all_selected = []
        total_acquired = 0
        if len(self.pool_indices) > 0 and total_budget_left > 0:
            if self.acquire_full_time:
                for _ in range(n_repeated_samples_acquire):
                    max_acquire = min(samples_acquire_batch_size, len(self.pool_indices), total_budget_left, samples_acquire_batch_size,
                                      int(np.ceil(total_budget_left / self.max_time)))
                    selected = self.acquire(model, k=max_acquire, batch_size=acquire_batch_size)
                    all_selected.append(selected)
                    n_acquired = self.__conduct_one_batch_full_time_acquire(selected, total_budget_left)
                    total_acquired += n_acquired
                    total_budget_left -= total_acquired
            else:
                for _ in range(n_repeated_samples_acquire):
                    max_acquire = min(samples_acquire_batch_size, len(self.pool_indices), total_budget_left, samples_acquire_batch_size)
                    selected = self.acquire(model, k=max_acquire, batch_size=acquire_batch_size)
                    all_selected.append(selected)
                    self.__conduct_one_batch_one_time_acquire(selected)
                    total_acquired += len(selected)
                    total_budget_left -= total_acquired
        return all_selected, total_acquired

    # --- Changes to run (bookkeeping/round_info updates) ---
    def run(self,
            rounds: int,
            samples_acquire_batch_size: int,
            n_repeated_samples_acquire: int,
            total_budget: int,
            max_epochs: int = 100,
            patience: int = 10,
            batch_size: int = 64,
            eval_batch_size: int = 2048,
            acquire_batch_size: int = 2048,
            lr: float = 1e-4,
            weight_decay: float = 0.0,
            # outdir: Path = Path('./al_runs'),
            retrain_freq: int = 1,
            tmp_save_dir='./tmp_save',
            saved_models_dir='./saved_models',
            last_round_epochs=200):
        tmp_save_dir = Path(tmp_save_dir)
        saved_models_dir = Path(saved_models_dir)
        total_budget_left = total_budget
        start_round = 1

        # Resume logic
        state_path = saved_models_dir / 'al_state_latest.pt'
        if state_path.exists():
            current_model, last_round = self.load_state(saved_models_dir)
            start_round = last_round + 1
            if start_round == rounds + 1:
                print(f"Found a complete run, skipping model fit..")
                return current_model
            total_acquired = self.compute_total_acquired(history_file_path=(saved_models_dir / 'al_history.json'))
            if total_acquired is None:
                raise Exception("stored total_acquired is None")

            total_budget_left -= total_acquired
            print(f"Found previous AL state. Resuming from round {last_round}..")
        else:
            current_model = self.model_class().to(self.device) if not self.retrain_from_scratch else None

        meta = {'retrain_from_scratch': self.retrain_from_scratch, 'acquisition': type(self.acquisition).__name__,
                'device': str(self.device)}
        (saved_models_dir / 'meta.json').write_text(json.dumps(meta, indent=2))
        for r in range(start_round, rounds + 1):
            if r == rounds:
                max_epochs = last_round_epochs
                do_tqdm = True
            else:
                do_tqdm = False
            print(
                f"\n=== AL Round {r}/{rounds} | labeled={len(self.labeled_indices)} | pool={len(self.pool_indices)} ===")
            if self.retrain_from_scratch or current_model is None:
                model = self.model_class()
                if self.verbose:
                    print("Retraining from scratch this round.")
            else:
                model = current_model
                if self.verbose:
                    print("Warm-starting from previous round's model.")
            os.makedirs(tmp_save_dir, exist_ok=True)
            ckpt_path = tmp_save_dir / f'round_{r:03d}_checkpoint.pt'
            model, train_losses, val_losses = self._fit_model(model, self.labeled_indices, self.val_indices,
                                                              lr=lr, max_epochs=max_epochs, patience=patience,
                                                              batch_size=batch_size, weight_decay=weight_decay,
                                                              eval_batch_size=eval_batch_size,
                                                              checkpoint_path=ckpt_path)
            # store model + state (including obs_lens and total acquisitions)

            # append epoch-wise losses
            self.history['training_losses'].append(train_losses)
            self.history['validation_losses'].append(val_losses)

            final_val = self._eval(model, self.val_indices, batch_size=eval_batch_size)

            # prepare round_info and additional bookkeeping required
            round_info = {
                'round': r,
                'n_labeled': len(self.labeled_indices),
                'n_pool': len(self.pool_indices),
                'final_val_loss': float(final_val),
                'acquired_indices': [],
                'train_losses': train_losses,  # epoch-wise training losses for this round
                'val_losses': val_losses  # epoch-wise validation losses for this round
            }

            # Acquire new labels with the stepwise reveal protocol

            all_selected, total_acquired = self.__conduct_one_batch_acquire(model, total_budget_left,
                                                                            acquire_batch_size,
                                                                            samples_acquire_batch_size,
                                                                            n_repeated_samples_acquire)
            total_budget_left -= total_acquired
            assert total_budget_left + self.total_acquisitions == total_budget

            if self.verbose:
                print(f"Acquired {total_acquired} new samples")

            round_info['acquired_indices'] = all_selected
            round_info['acquired_count'] = total_acquired

                # update cumulative acquisitions

            if not self.retrain_from_scratch:
                current_model = model

            # Count events observed and record observed timesteps per index
            # observed_timesteps: store as {str(idx): obs_len} to be JSON-serializable keys
            observed_map = {str(idx): int(self.dataset.obs_lens[idx]) for idx in range(len(self.dataset))}
            events_count = self._count_events_observed()

            round_info['observed_timesteps'] = observed_map
            round_info['events_observed'] = int(events_count)
            round_info['total_acquisitions'] = int(self.total_acquisitions)

            # append to history
            self.history['rounds'].append(round_info)
            self.history['acquisition_sizes'].append(len(round_info.get('acquired_indices', [])))
            self.history['events_observed'].append(int(events_count))
            self.history['observed_timesteps'].append(observed_map)

            (tmp_save_dir / f'history_round_{r:03d}.json').write_text(json.dumps(round_info, indent=2))
            self.store_state(model, saved_models_dir, r)
            self._plot_history(tmp_save_dir)

            (saved_models_dir / 'al_history.json').write_text(json.dumps(self.history, indent=2))
        print(f"AL run complete. Results saved to {saved_models_dir}")
        return model

    def _plot_history(self, outdir: Path):
        plt.figure(figsize=(10, 6))
        for r, (tr, va) in enumerate(zip(self.history['training_losses'], self.history['validation_losses']), start=1):
            epochs = list(range(1, len(tr) + 1))
            offset = sum(len(x) for x in self.history['training_losses'][:r - 1])
            plt.plot([offset + e for e in epochs], tr, alpha=0.6, label=f'R{r} train' if r == 1 else None)
            plt.plot([offset + e for e in epochs], va, alpha=0.9, linestyle='--', label=f'R{r} val' if r == 1 else None)
        acc_offsets = []
        offset = 0
        for tr in self.history['training_losses']:
            offset += len(tr)
            acc_offsets.append(offset)
        for ao in acc_offsets:
            plt.axvline(x=ao, color='gray', linestyle=':', linewidth=0.8)
        plt.xlabel('Epochs (concatenated across rounds)')
        plt.ylabel('Loss')
        plt.title('Training & Validation Losses across AL rounds')
        plt.legend()
        plt.grid(True, alpha=0.2)
        p = outdir / 'loss_curve.png'
        plt.savefig(p, bbox_inches='tight')
        plt.close()

        for r, (tr, va) in enumerate(zip(self.history['training_losses'], self.history['validation_losses']), start=1):
            plt.figure(figsize=(6, 4))
            plt.plot(range(1, len(tr) + 1), tr, label='train')
            plt.plot(range(1, len(va) + 1), va, linestyle='--', label='val')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Round {r} losses')
            plt.legend()
            plt.grid(True, alpha=0.2)
            plt.savefig(outdir / f'loss_round_{r:03d}.png', bbox_inches='tight')
            plt.close()
