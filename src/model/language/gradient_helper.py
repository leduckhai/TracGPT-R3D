import os
import torch 
class GradientTracker:
    def __init__(self, gradient_path, log_gradients=False):
        self.gradient_path = gradient_path
        self.log_gradients = log_gradients
        self.gradients = []

    def _init_gradient_logging(self):
        """Initialize gradient log file (overwrites existing)"""
        try:
            os.makedirs(os.path.dirname(self.gradient_path), exist_ok=True)
            
            with open(self.gradient_path, 'w') as f:
                headers = [
                    f"{'Step':<8}",
                    f"{'Layer':<40}",
                    f"{'Mean':>12}",
                    f"{'Max':>12}",
                    f"{'Std':>12}",
                    f"{'Param/Mean':>12}",
                    f"{'NaN%':>6}"
                ]
                f.write(" ".join(headers) + "\n")
                f.write("-" * 100 + "\n")
                
            self.gradient_buffer = []
            self.buffer_size = 20
            
        except Exception as e:
            print(f"⚠️ Failed to initialize gradient logging: {str(e)}")
            self.log_gradients = False
    def _check_gradients(self):
        """Detect NaN/exploding gradients before optimizer step"""
        nan_found = False
        explode_found = False
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"⚠️ NaN gradients in {name}!")
                    nan_found = True
                if (param.grad.abs() > 1e3).any():
                    print(f"🚨 Exploding gradients in {name} (max={param.grad.abs().max():.1e})")
                    explode_found = True
                    
        return nan_found or explode_found

    def _clip_gradients(self):
            """Layer-wise gradient clipping with logging"""
            global_norms = {}
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if "bbox_head" in name:
                        global_norms[name] = 1.0  # Tighter clipping
                    elif "vision_tower" in name:
                        global_norms[name] = 5.0  # Looser clipping
                    else:
                        global_norms[name] = 2.0  # Default

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    torch.nn.utils.clip_grad_norm_(
                        [param], 
                        max_norm=global_norms[name],
                        norm_type=2
                    )
    def _log_gradient_stats(self, model, step):
        print("Logging gradient stats...")
        print(f"Writing to: {os.path.abspath(self.gradient_path)}")  # Check full path
        print(f"File exists: {os.path.exists(self.gradient_path)}")  # False = path issue
        try:
            # Open file in append mode
            with open(self.gradient_path, 'a') as f:
                f.write(f"-------Global step: {self.state.global_step:<8}\n")
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad = param.grad.detach()
                        param_data = param.detach()
                        
                        nan_count = torch.isnan(grad).sum().item()
                        total_elements = grad.numel()
                        
                        if nan_count == total_elements:
                            stats = f"{name:<40} {'NaN':>12} {'NaN':>12} {'NaN':>12} {'NaN':>12} {100.0:>6.1f}%"
                        else:
                            grad_mean = grad[~torch.isnan(grad)].mean().item()
                            grad_max = grad[~torch.isnan(grad)].max().item()
                            grad_std = grad[~torch.isnan(grad)].std().item()
                            param_ratio = abs(param_data.mean().item() / (grad_mean + 1e-9))
                            nan_pct = (nan_count / total_elements) * 100
                            
                            stats = (
                                f"{name:<40} "
                                f"{grad_mean:>12.4e} "
                                f"{grad_max:>12.4e} "
                                f"{grad_std:>12.4e} "
                                f"{param_ratio:>12.2f} "
                                f"{nan_pct:>6.1f}%"
                            )
                        
                        f.write(f"{stats}\n")

                f.flush()

        except Exception as e:
            print(f"⚠️ Failed to log gradients at step {step}: {str(e)}")
            if torch.cuda.is_available():
                print(f"Current GPU memory: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
    def _log_gradients(self, model):
        """
        Log gradient statistics to wandb
        """
        if wandb.run is None:
            return
            
        gradient_logs = {}
        total_norm = 0.0
        param_count = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None and param.requires_grad:
                # Gradient norms
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
                
                # Log statistics
                prefix = f"gradients/{name.replace('.', '/')}"
                gradient_logs.update({
                    f"{prefix}/norm": param_norm.item(),
                    f"{prefix}/mean": param.grad.mean().item(),
                    f"{prefix}/std": param.grad.std().item(),
                    f"{prefix}/max": param.grad.max().item(),
                    f"{prefix}/min": param.grad.min().item(),
                })
                
                # Histograms
                wandb.log({
                    f"grad_hist/{name}": wandb.Histogram(param.grad.cpu().numpy())
                }, step=self.state.global_step)
        
        # Log total gradient statistics
        if param_count > 0:
            total_norm = total_norm ** 0.5
            gradient_logs.update({
                "gradients/total_norm": total_norm,
                "gradients/avg_norm": total_norm / param_count
            })
            
            wandb.log(gradient_logs, step=self.state.global_step)
    
def print_gradient_stats_to_file(model, step, file_path, print_console=False):
    """Log gradient statistics to a file with proper formatting.
    
    Args:
        model: PyTorch model
        step: Training step/epoch
        file_path: Output file path
        print_console: Whether to also print to console
    """
    header = f"\n{'Step':<8} {'Layer':<30} {'Mean':>10} {'Max':>10} {'Std':>10}\n"
    header += "-" * 68
    
    lines = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            line = (f"{step:<8} {name:<30} "
                   f"{param.grad.mean().item():>10.4f} "
                   f"{param.grad.max().item():>10.4f} "
                   f"{param.grad.std().item():>10.4f}")
            lines.append(line)
    
    with open(file_path, 'a') as f:
        if step == 0:
            f.write(header + "\n")
        
        f.write("\n".join(lines) + "\n")
    
    if print_console:
        print(header)
        print("\n".join(lines))
