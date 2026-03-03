import os
import shutil

def nuke_finlab_cache():
    print("===========================================")
    print("🔥 啟動 FinLab 有毒快取 (Pickle) 強制清除程式 🔥")
    print("===========================================")

    # 預設路徑 (適用於 Windows/Mac/Linux)
    home_dir = os.path.expanduser("~")
    finlab_cache_dir = os.path.join(home_dir, ".finlab")

    if os.path.exists(finlab_cache_dir):
        print(f"⚠️ 偵測到 FinLab 快取資料夾: {finlab_cache_dir}")
        print("⏳ 正在強制刪除中...")
        try:
            shutil.rmtree(finlab_cache_dir)
            print("✅ 刪除成功！所有舊版 Pandas 留下來的「有毒快取」已被清空。")
        except Exception as e:
            print(f"❌ 刪除失敗，請確認該資料夾沒有被其他程式佔用。錯誤訊息: {e}")
    else:
        print(f"✅ 未發現 FinLab 快取資料夾 ({finlab_cache_dir})，您的環境很乾淨！")

    print("\n💡 請回到專案目錄，重新執行 streamlit run app.py，系統會自動使用相容的新版 Pandas 下載乾淨的資料。")

if __name__ == "__main__":
    nuke_finlab_cache()
