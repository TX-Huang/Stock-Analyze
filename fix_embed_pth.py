import os
import glob
import sys

def fix_pth_file():
    """
    Finds the pythonXX._pth file in the python_embed directory
    and uncomments 'import site' to enable site-packages support.
    """
    embed_dir = os.path.join(os.getcwd(), 'python_embed')
    if not os.path.exists(embed_dir):
        print("[Fix] python_embed directory not found. Skipping.")
        return

    # Find the ._pth file (e.g., python311._pth)
    pth_files = glob.glob(os.path.join(embed_dir, 'python*._pth'))

    if not pth_files:
        print("[Fix] No ._pth file found in python_embed.")
        return

    pth_file = pth_files[0]
    print(f"[Fix] Found configuration file: {pth_file}")

    try:
        with open(pth_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        new_lines = []
        fixed = False
        for line in lines:
            # Uncomment 'import site'
            if line.strip() == '#import site':
                new_lines.append('import site\n')
                fixed = True
                print("[Fix] Uncommented 'import site'.")
            else:
                new_lines.append(line)

        if fixed:
            with open(pth_file, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            print("[Fix] Configuration updated successfully.")
        else:
            print("[Fix] 'import site' already enabled or not found.")

    except Exception as e:
        print(f"[Error] Failed to patch ._pth file: {e}")

if __name__ == "__main__":
    fix_pth_file()
