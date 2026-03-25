import copy
import numpy as np
import torch
import torch.optim as optim


class BaseNetTrainingMixin:
    def _evaluate_current_loss(self, model, data_GPU, kwargs) -> float:
        loss = model.physics_loss(data_GPU, **kwargs)
        return float(loss.detach().item())

    def _snapshot_model_state(self, model):
        return copy.deepcopy(model.state_dict())

    def train_net(self, data, model, data_GPU, **kwargs):
        max_retries = getattr(self.config, "max_retries", 1)
        retry_count = 0
        best_loss = float("inf")
        best_model = None
        dt = kwargs.get("dt", 1.0)
        tolerance = self._get_training_tolerance(dt)

        while retry_count <= max_retries:
            if retry_count > 0:
                self._reset_random_seed_for_retry(retry_count)
                self._reinitialize_model_for_retry(model)

            optimizer_adam, optimizer_lbfgs = self._build_optimizers(model)
            adjust_learning_rate = self._build_lr_adjuster(optimizer_adam)
            model = model.to(self.config.device)
            final_loss, run_best_loss, run_best_model = self._run_optimizer_sequence(
                model,
                data_GPU,
                kwargs,
                optimizer_adam,
                optimizer_lbfgs,
                adjust_learning_rate,
                tolerance,
            )

            if run_best_loss < best_loss and run_best_model is not None:
                best_loss = run_best_loss
                best_model = run_best_model

            if run_best_loss < tolerance:
                print(
                    f"Target accuracy {tolerance:.2e} "
                    f"(DNNtol={self.config.DNNtol:.2e}) achieved, stopping retries"
                )
                break
            retry_count += 1

        if best_model is not None:
            model.load_state_dict(best_model)
            print(f"Using best model, loss value: {best_loss}")
        return model

    def _get_training_tolerance(self, dt: float) -> float:
        _ = dt
        return float(self.config.DNNtol)

    def _reset_random_seed_for_retry(self, retry_count: int) -> None:
        print(
            f"Retry {retry_count}/{getattr(self.config, 'max_retries', 1)}, "
            f"changing random seed to {self.config.seed + retry_count}"
        )
        torch.manual_seed(self.config.seed + retry_count)
        np.random.seed(self.config.seed + retry_count)

    def _reinitialize_model_for_retry(self, model) -> None:
        if hasattr(model.__class__, "model_init"):
            fresh_model = model.__class__.model_init(self.config).to(self.config.device)
            model.load_state_dict(fresh_model.state_dict())
            return

        for module in model.modules():
            reset_fn = getattr(module, "reset_parameters", None)
            if callable(reset_fn):
                reset_fn()

    def _build_optimizers(self, model):
        adam_betas = getattr(self.config, "adam_betas", None)
        if adam_betas is None:
            optimizer_adam = optim.Adam(
                model.parameters(),
                lr=self.config.learning_rate,
            )
        else:
            optimizer_adam = optim.Adam(
                model.parameters(),
                lr=self.config.learning_rate,
                betas=tuple(adam_betas),
            )
        optimizer_lbfgs = None
        if getattr(self.config, "epochs_lbfgs", 0) > 0:
            optimizer_lbfgs = optim.LBFGS(model.parameters(), lr=0.1)
        return optimizer_adam, optimizer_lbfgs

    def _build_lr_adjuster(self, optimizer):
        if not getattr(self.config, "adaptive_learning_rate", False):
            def identity_adjuster(current_loss, epoch):
                _ = (current_loss, epoch)
                return False, optimizer.param_groups[0]["lr"]

            return identity_adjuster

        lr_thresholds = [1.0, 0.5, 1e-2, 1e-3, 1e-4]
        lr_values = [self.config.learning_rate, 0.005, 0.003, 0.001, 0.0005, 0.0001]
        lr_history = []

        def adjust_learning_rate(current_loss, epoch):
            _ = epoch
            lr_history.append(current_loss)
            current_lr = optimizer.param_groups[0]["lr"]
            target_lr = lr_values[0]
            for index, threshold in enumerate(lr_thresholds):
                if current_loss < threshold:
                    target_lr = lr_values[index + 1]
            target_lr = min(target_lr, current_lr)
            if abs(current_lr - target_lr) > 1e-8:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = target_lr
                return True, target_lr
            return False, current_lr

        return adjust_learning_rate

    def _run_optimizer_sequence(
        self,
        model,
        data_GPU,
        kwargs,
        optimizer_adam,
        optimizer_lbfgs,
        adjust_learning_rate,
        tolerance: float,
    ):
        def closure():
            optimizer_adam.zero_grad()
            loss = model.physics_loss(data_GPU, **kwargs)
            loss.backward()
            return loss

        final_loss, best_loss, best_model = self._run_adam_phase(
            model,
            data_GPU,
            kwargs,
            optimizer_adam,
            adjust_learning_rate,
            tolerance,
        )
        print(f"Final Adam Loss: {final_loss}")

        if self.config.epochs_lbfgs > 0 and optimizer_lbfgs is not None:
            final_loss, best_loss, best_model = self._run_lbfgs_phase(
                model,
                data_GPU,
                kwargs,
                optimizer_lbfgs,
                closure,
                tolerance,
                final_loss,
                best_loss,
                best_model,
            )
        return final_loss, best_loss, best_model

    def _run_adam_phase(self, model, data_GPU, kwargs, optimizer_adam, adjust_learning_rate, tolerance: float):
        final_loss = None
        best_loss = float("inf")
        best_model = None
        for epoch in range(self.config.epochs_adam):
            optimizer_adam.zero_grad()
            loss = model.physics_loss(data_GPU, **kwargs)
            loss.backward()
            optimizer_adam.step()
            final_loss = self._evaluate_current_loss(model, data_GPU, kwargs)
            if final_loss < best_loss:
                best_loss = final_loss
                best_model = self._snapshot_model_state(model)
            lr_changed, current_lr = adjust_learning_rate(final_loss, epoch)
            if final_loss < tolerance:
                break
            if epoch % 1000 == 0:
                print(f" Adam Epoch {epoch}, Loss: {final_loss:.6e}, LR: {current_lr:.6e}")
            if lr_changed:
                print(f"   Learning rate adjusted to {current_lr:.6e} at epoch {epoch}, loss {final_loss:.6e}")
        return final_loss, best_loss, best_model

    def _run_lbfgs_phase(
        self,
        model,
        data_GPU,
        kwargs,
        optimizer_lbfgs,
        closure,
        tolerance: float,
        final_loss: float,
        best_loss: float,
        best_model,
    ):
        for epoch in range(self.config.epochs_lbfgs):
            optimizer_lbfgs.zero_grad()
            loss = model.physics_loss(data_GPU, **kwargs)
            loss.backward()
            optimizer_lbfgs.step(closure)
            final_loss = self._evaluate_current_loss(model, data_GPU, kwargs)
            if final_loss < best_loss:
                best_loss = final_loss
                best_model = self._snapshot_model_state(model)
            if final_loss < tolerance:
                break
            if epoch % 100 == 0:
                print(f"LBFGS Epoch {epoch}, Loss: {final_loss}")
        return final_loss, best_loss, best_model
