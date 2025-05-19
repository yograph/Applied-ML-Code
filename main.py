import os
import sys
import subprocess
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Cancer_Detection.data.processing_to_png import DicomToPngConverter
from Cancer_Detection.models.main_model import FullTrainingOfModel

def default_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RunningCode:
    """
    Modes:
      - Changing_Image_Type: convert DICOMs to PNGs.
      - Submitting_to_testing: load saved model and run predictions on lockbox samples.
      - Training: launch YOLOX training via subprocess.
      - Local_Run: local model training, plotting, and evaluation.
    """
    def __init__(self, args):
        self.device = default_device()
        self.class_names = args.class_names or []
        self.save_dir = args.save_dir
        datasets = {}
        # Collect dataset args
        if args.input_dir: datasets['input_dir'] = args.input_dir
        if args.output_dir: datasets['output_dir'] = args.output_dir
        if args.lockbox_dir: datasets['lockbox_loader'] = self._make_loader(args.lockbox_dir, args.batch_size)

        # Prepare DataLoaders for ConvNet
        train_ds, val_ds, test_ds = None, None, None
        if args.train_dir:
            train_ds = self._make_loader(args.train_dir, args.batch_size, labels=True)
        if args.val_dir:
            val_ds = self._make_loader(args.val_dir, args.batch_size, labels=True)
        if args.test_dir:
            test_ds = self._make_loader(args.test_dir, args.batch_size, labels=True)

        # Initialize trainer for our ConvNet
        self.trainer = FullTrainingOfModel(
            train_ds, val_ds, test_ds,
            batch_size=args.batch_size,
            lr=args.lr,
            num_epochs=args.num_epochs,
            device=self.device,
            class_names=self.class_names
        )
        self.trainer.initialize_models(num_models=1)

        # Build YOLOX command for distributed training
        self.cmd = [
            sys.executable,
            os.path.join(os.getcwd(), 'Cancer_Detection', 'data', 'YOLOX', 'tools', 'train.py'),
            '--num_gpus', str(args.num_gpus),
            '--num_machines', str(args.num_machines),
            '--machine_rank', str(args.machine_rank),
            '--dist_backend', args.dist_backend
        ]
        if args.dist_url:
            self.cmd += ['--dist_url', args.dist_url]
        if args.yolo_args:
            self.cmd += args.yolo_args

        # Dispatch mode
        mode = args.mode
        if mode == 'Changing_Image_Type':
            converter = DicomToPngConverter(
                datasets['input_dir'], datasets['output_dir'],
                resize_to=args.resize, num_workers=args.num_workers)
            converter.run()

        elif mode == 'Submitting_to_testing':
            model_name = 'model_0'
            weight_path = os.path.join(self.save_dir, f"{model_name}_weights.pth")
            self.trainer.load_model(model_name, weight_path)
            loader = datasets.get('lockbox_loader')
            if loader is None:
                raise ValueError("--lockbox_dir required for Submitting_to_testing mode")
            model = self.trainer.models[model_name]
            model.eval()
            preds = []
            with torch.no_grad():
                for imgs in loader:
                    imgs = imgs.to(self.device)
                    out = model(imgs)
                    p = nn.functional.softmax(out, dim=1).argmax(dim=1)
                    preds.extend(p.cpu().tolist())
            print('Lockbox predictions:', preds)

        elif mode == 'Training':
            print(f"Running YOLOX training: {' '.join(self.cmd)}")
            res = subprocess.run(self.cmd, capture_output=True, text=True)
            print(res.stdout)
            print(res.stderr)

        elif mode == 'Local_Run':
            self.trainer.train()
            self.trainer.plot_all_loss_curves()
            self.eval_results = self.trainer.evaluate_models()

        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _make_loader(self, directory, batch_size, labels=False):
        """Utility to create a DataLoader from directory of images (+ labels if needed)."""
        # Placeholder: replace with actual Dataset class
        from torchvision import datasets, transforms
        transform = transforms.Compose([transforms.ToTensor()])
        ds = datasets.ImageFolder(directory, transform=transform) if labels else datasets.ImageFolder(directory, transform=transform)
        return DataLoader(ds, batch_size=batch_size, shuffle=labels)


def parse_args():
    parser = argparse.ArgumentParser(description='Run Cancer Detection Pipeline')
    parser.add_argument('mode', choices=['Changing_Image_Type','Submitting_to_testing','Training','Local_Run'], help='Operation mode')
    # Dataset paths
    parser.add_argument('--input_dir', type=str, help='DICOM input directory')
    parser.add_argument('--output_dir', type=str, help='PNG output directory')
    parser.add_argument('--lockbox_dir', type=str, help='Directory for lockbox samples')
    parser.add_argument('--train_dir', type=str, help='Training images dir')
    parser.add_argument('--val_dir', type=str, help='Validation images dir')
    parser.add_argument('--test_dir', type=str, help='Test images dir')
    # Training hyperparams
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--resize', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default='model_weights/')
    parser.add_argument('--class_names', nargs='+', help='List of class names')
    # YOLOX distributed args
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--num_machines', type=int, default=1)
    parser.add_argument('--machine_rank', type=int, default=0)
    parser.add_argument('--dist_backend', type=str, default='nccl')
    parser.add_argument('--dist_url', type=str, help='URL for distributed backend')
    parser.add_argument('--yolo_args', nargs=argparse.REMAINDER, help='Additional args for YOLOX')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    RunningCode(args)
