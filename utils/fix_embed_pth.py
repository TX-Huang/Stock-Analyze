import os
import glob

def fix_pth_file():
    """
    Finds the pythonXX._pth file in the python_embed directory
    and uncomments 'import site' and adds 'Lib\\site-packages'
    to enable proper pip support.
    """
    embed_dir = os.path.join(os.getcwd(), 'python_embed')
    if not os.path.exists(embed_dir):
        print("[Fix] python_embed directory not found. Skipping.")
        return

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
        has_site = False
        has_lib = False

        for line in lines:
            clean = line.strip()
            if clean == '#import site':
                new_lines.append('import site\n')
                has_site = True
            elif clean == 'import site':
                new_lines.append(line)
                has_site = True
            elif clean == 'Lib\\site-packages' or clean == './Lib/site-packages':
                new_lines.append(line)
                has_lib = True
            else:
                new_lines.append(line)

        if not has_lib:
            # Add before import site if possible, else at end
            if has_site and 'import site\n' in new_lines:
                idx = new_lines.index('import site\n')
                new_lines.insert(idx, 'Lib\\site-packages\n')
            else:
                new_lines.append('Lib\\site-packages\n')

        with open(pth_file, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)

        print("[Fix] Configuration updated successfully (added Lib\\site-packages and enabled site).")

    except Exception as e:
        print(f"[Error] Failed to patch ._pth file: {e}")

if __name__ == "__main__":
    fix_pth_file()
