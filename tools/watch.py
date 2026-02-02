import os, time
import matplotlib.pyplot as plt
import numpy as np


def smooth(y, window_size=50):
    box = np.ones(window_size)/window_size
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


class TrainingVisualizer:
    def __init__(self, save_dir='log/training_plots'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.steps = []
        self.meta_losses = []  
        self.ce_losses = []   

    def update(self, step, meta_loss, ce_loss):
        # 记录每一步的标量
        self.steps.append(step)
        self.meta_losses.append(meta_loss)
        self.ce_losses.append(ce_loss)

    def plot(self):
        if len(self.steps) < 2: return

        plt.figure(figsize=(14, 6))
        
        # 时序演变
        plt.subplot(1, 2, 1)
      
        plt.plot(self.steps, self.meta_losses, alpha=0.2, color='blue')
        plt.plot(self.steps, smooth(self.meta_losses), label='Total Loss (Meta)', color='blue')
        
        plt.plot(self.steps, self.ce_losses, alpha=0.2, color='orange')
        plt.plot(self.steps, smooth(self.ce_losses), label='Real CE Loss (GT)', color='orange')
        
        plt.title('Loss Evolution: Meta-Cognition vs Reality')
        plt.xlabel('Steps')
        plt.ylabel('Loss Value')
        plt.ylim(0, 5.5)  
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 散点相关性性
        plt.subplot(1, 2, 2)
        # 最近1000个点
        ce_last = self.ce_losses[-1000:]
        meta_last = self.meta_losses[-1000:]
        
        plt.scatter(ce_last, meta_last, alpha=0.5, s=10, c=self.steps[-1000:], cmap='viridis')
        
        # y = x^2 的参考线
        if ce_last: 
            x_min, x_max = min(ce_last), max(ce_last)
            x_fit = np.linspace(x_min, x_max, 100)
            y_fit = x_fit**2
            plt.plot(x_fit, y_fit, 'r--', label='y = x²', linewidth=2, alpha=0.8)
        
        plt.title('Correlation: Real Error vs Meta Loss (Last 1k steps)')
        plt.xlabel('Real CE Loss (Difficulty)')
        plt.ylabel('Total Meta Loss')
        plt.colorbar(label='Step Age')
        plt.legend() 
        plt.grid(True, alpha=0.3)
        
        # 保存图片
        timestamp = int(time.time())
        save_path = os.path.join(self.save_dir, f"training_monitor_{timestamp}.png")
        plt.savefig(save_path)
        plt.close()