import torch
import os


# En son kaydedilen checkpoint dosyasının tam yolunu buraya yaz veya
# runs/ppo_goal klasöründeki bir dosyayı otomatik bulalım:
def check_latest_checkpoint(folder="runs/quick_test"):
    if not os.path.exists(folder):
        print(f"Klasör bulunamadı: {folder}")
        return

    files = [f for f in os.listdir(folder) if f.endswith(".pt")]
    if not files:
        print("Klasörde .pt dosyası yok.")
        return

    # En son dosyayı al
    latest_file = max([os.path.join(folder, f) for f in files], key=os.path.getctime)
    print(f"İncelenen dosya: {latest_file}\n")

    try:
        # Dosyayı yükle (CPU'ya map ederek)
        checkpoint = torch.load(latest_file, map_location=torch.device('cpu'))

        print("✅ Dosya başarıyla yüklendi! Bozuk değil.")
        print("-" * 30)
        print("İçerik Anahtarları (Keys):")
        for key in checkpoint.keys():
            print(f"- {key}")

        if 'steps' in checkpoint:
            print(f"\nKaydedilen Adım Sayısı: {checkpoint['steps']}")

        if 'ac_state_dict' in checkpoint:
            print("Model ağırlıkları (state_dict) mevcut.")

    except Exception as e:
        print("❌ Dosya yüklenirken hata oluştu!")
        print(f"Hata detayı: {e}")


if __name__ == "__main__":
    check_ckpt_path = "check_ckpt.py"
    check_latest_checkpoint()