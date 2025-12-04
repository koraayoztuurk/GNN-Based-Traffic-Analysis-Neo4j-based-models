#!/usr/bin/env python3
"""
incremental_train.py
--------------------
Incremental (artımlı) training scripti

Yeni veri geldiğinde mevcut modeli fine-tune eder:
1. En son checkpoint'i yükle
2. Sadece yeni verileri çek (son N gün)
3. Mevcut model üzerine eğit (fine-tune)
4. Yeni checkpoint kaydet

Kullanım:
    # Son 7 gün verisi ile fine-tune
    python incremental_train.py --last_n_days 7
    
    # Belirli tarih aralığı
    python incremental_train.py --start_time 2024-11-20T00:00:00Z
    
    # Pipeline ile entegrasyon
    python run_pipeline.py  # Yeni veri çek
    python incremental_train.py --last_n_days 1  # Son 1 gün ile eğit
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import argparse

import torch


def find_latest_checkpoint(checkpoint_dir: str = "outputs/models") -> str:
    """En son checkpoint'i bul"""
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return None
    
    # best_model.pt varsa onu kullan
    best_model = checkpoint_dir / "best_model.pt"
    if best_model.exists():
        return str(best_model)
    
    # Yoksa en son checkpoint_epoch_*.pt'yi bul
    checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    if not checkpoints:
        return None
    
    # Epoch numarasına göre sırala
    checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]), reverse=True)
    return str(checkpoints[0])


def main():
    parser = argparse.ArgumentParser(description='Incremental Training - Yeni veriyle model güncelle')
    
    # Data filtering (en az biri gerekli)
    parser.add_argument('--last_n_days', type=int, default=None,
                        help='Son N gün verisi kullan (örn: 7)')
    parser.add_argument('--start_time', type=str, default=None,
                        help='Başlangıç zamanı (ISO: 2024-11-20T00:00:00Z)')
    parser.add_argument('--end_time', type=str, default=None,
                        help='Bitiş zamanı (ISO: 2024-11-27T23:59:59Z)')
    
    # Training params (default: hızlı fine-tune)
    parser.add_argument('--epochs', type=int, default=20,
                        help='Epoch sayısı (default: 20, fine-tune için yeterli)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate (default: 0.0001, düşük = fine-tune)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    
    # Checkpoint
    parser.add_argument('--checkpoint_dir', type=str, default='outputs/models',
                        help='Checkpoint dizini')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Kullanılacak checkpoint (None = otomatik bul)')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Validation
    if not any([args.last_n_days, args.start_time, args.end_time]):
        print("❌ Hata: En az bir filtre gerekli!")
        print("   --last_n_days VEYA --start_time/--end_time kullanın")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("🔄 Incremental Training - Model Güncelleme")
    print("="*70 + "\n")
    
    # 1. Checkpoint bul
    checkpoint_path = args.checkpoint or find_latest_checkpoint(args.checkpoint_dir)
    
    if not checkpoint_path:
        print("❌ Checkpoint bulunamadı!")
        print(f"   Dizin: {args.checkpoint_dir}")
        print("\n💡 İlk önce tam training yapın:")
        print("   python src/gnn/train.py --epochs 100")
        sys.exit(1)
    
    print(f"📦 Checkpoint bulundu: {checkpoint_path}")
    
    # Checkpoint bilgisi
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    epoch = checkpoint.get('epoch', '?')
    val_loss = checkpoint.get('best_val_loss', '?')
    print(f"  - Epoch: {epoch}")
    print(f"  - Best val loss: {val_loss}")
    
    # 2. Tarih aralığını hesapla
    if args.last_n_days:
        # Son N gün
        end_time = datetime.now().isoformat() + "Z"
        start_time = (datetime.now() - timedelta(days=args.last_n_days)).isoformat() + "Z"
        print(f"\n📅 Son {args.last_n_days} gün verisi:")
        print(f"  {start_time} → {end_time}")
    else:
        start_time = args.start_time
        end_time = args.end_time
        print(f"\n📅 Belirli tarih aralığı:")
        print(f"  {start_time or 'başlangıç'} → {end_time or 'şimdi'}")
    
    # 3. train.py'yi çağır (fine-tune mode)
    print(f"\n🚀 Fine-tuning başlıyor...")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Learning rate: {args.lr} (düşük = fine-tune)")
    print(f"  - Batch size: {args.batch_size}")
    print()
    
    # train.py komutunu oluştur
    train_cmd = [
        sys.executable,
        "src/gnn/train.py",
        "--resume", checkpoint_path,
        "--fine_tune",
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--batch_size", str(args.batch_size),
        "--device", args.device,
        "--checkpoint_dir", args.checkpoint_dir
    ]
    
    # Timestamp filtreleri ekle
    if args.last_n_days:
        train_cmd.extend(["--use_last_n_days", str(args.last_n_days)])
    else:
        if start_time:
            train_cmd.extend(["--start_time", start_time])
        if end_time:
            train_cmd.extend(["--end_time", end_time])
    
    # Çalıştır
    import subprocess
    result = subprocess.run(train_cmd)
    
    if result.returncode == 0:
        print("\n✅ Incremental training tamamlandı!")
        print(f"📁 Model: {args.checkpoint_dir}/best_model.pt")
    else:
        print("\n❌ Training başarısız!")
        sys.exit(1)


if __name__ == "__main__":
    main()
