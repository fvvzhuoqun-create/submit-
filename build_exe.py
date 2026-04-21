import PyInstaller.__main__

if __name__ == '__main__':
    PyInstaller.__main__.run([
        'gui_app.py',
        '--noconsole',
        '--onedir',
        '--hidden-import=tkinter',
        '--hidden-import=tkinter.messagebox',
        '--hidden-import=pandas',
        '--hidden-import=openpyxl',
        '--hidden-import=huggingface_hub',
        '--hidden-import=torch',
        '--hidden-import=torch_geometric'
    ])