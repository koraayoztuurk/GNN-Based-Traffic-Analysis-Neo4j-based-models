#!/usr/bin/env python3
"""
check_model.py
--------------
Eğitilmiş model checkpoint'lerini kontrol eder.
Hem STA-GCN hem de STGMS modellerini destekler.
"""

import torch
from pathlib import Path
import sys


def check_checkpoint(model_path: str, model_name: str):
    """Checkpoint dosyasını kontrol et ve bilgileri göster"""
    path = Path(model_path)
    
    if not path.exists():
        print(f"❌ {model_name}: Dosya bulunamadı - {model_path}")
        return False
    
    print(f"\n{'='*70}")
    print(f"📦 {model_name} Checkpoint Bilgileri")
    print(f"{'='*70}")
    print(f"📂 Dosya: {model_path}")
    print(f"📏 Boyut: {path.stat().st_size / (1024*1024):.2f} MB")
    
    try:
        ckpt = torch.load(model_path, map_location='cpu')
        
        # Temel bilgiler
        print(f"\n🔑 Checkpoint Anahtarları:")
        for key in ckpt.keys():
            print(f"   - {key}")
        
        # Epoch bilgisi
        epoch = ckpt.get('epoch', '?')
        print(f"\n📊 Eğitim Bilgileri:")
        print(f"   Epoch: {epoch}")
        
        # Loss bilgileri
        if 'train_loss' in ckpt:
            print(f"   Train Loss: {ckpt['train_loss']:.6f}")
        if 'val_loss' in ckpt:
            print(f"   Val Loss: {ckpt['val_loss']:.6f}")
        if 'best_val_loss' in ckpt:
            print(f"   Best Val Loss: {ckpt['best_val_loss']:.6f}")
        
        # Metrik bilgileri
        metrics = ['val_mae', 'val_rmse', 'val_mape', 'val_r2']
        found_metrics = {m: ckpt.get(m) for m in metrics if m in ckpt}
        if found_metrics:
            print(f"\n📈 Performans Metrikleri:")
            for metric, value in found_metrics.items():
                print(f"   {metric.upper()}: {value:.6f}")
        
        # Model state bilgisi
        if 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
            num_params = sum(p.numel() for p in state_dict.values())
            print(f"\n🏗️  Model Yapısı:")
            print(f"   Toplam Parametre: {num_params:,}")
            print(f"   Layer Sayısı: {len(state_dict)}")
            
            # İlk birkaç layer ismini göster
            layer_names = list(state_dict.keys())[:5]
            print(f"\n   İlk Katmanlar:")
            for name in layer_names:
                shape = tuple(state_dict[name].shape)
                print(f"      - {name}: {shape}")
            if len(state_dict) > 5:
                print(f"      ... (+{len(state_dict)-5} katman daha)")
        
        # Optimizer bilgisi
        if 'optimizer_state_dict' in ckpt:
            print(f"\n⚙️  Optimizer: Kaydedildi ✓")
        
        print(f"\n✅ Checkpoint başarıyla yüklendi!")
        return True
        
    except Exception as e:
        print(f"\n❌ Hata: {e}")
        return False


def main():
    """Ana fonksiyon - tüm modelleri kontrol et"""
    print("\n" + "="*70)
    print("🔍 MODEL CHECKPOINT KONTROLÜ")
    print("="*70)
    
    models_dir = Path("outputs/models")
    
    # Kontrol edilecek modeller
    model_files = {
        "STA-GCN (best)": "outputs/models/best_model.pt",
        "STA-GCN (latest)": "outputs/models/checkpoint_latest.pt",
        "STGMS (best)": "outputs/models/stgms_best.pt",
        "STGMS (latest)": "outputs/models/stgms_latest.pt",
    }
    
    found_any = False
    for model_name, model_path in model_files.items():
        if Path(model_path).exists():
            check_checkpoint(model_path, model_name)
            found_any = True
    
    # Hiçbir model bulunamadıysa
    if not found_any:
        print(f"\n❌ Hiçbir model checkpoint'i bulunamadı!")
        print(f"\n📂 Aranan klasör: {models_dir.absolute()}")
        
        if models_dir.exists():
            files = list(models_dir.glob("*.pt"))
            if files:
                print(f"\n💡 Bulunan .pt dosyaları:")
                for f in files:
                    print(f"   - {f.name}")
            else:
                print(f"\n📭 Klasör boş.")
        else:
            print(f"\n📭 Klasör mevcut değil.")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
