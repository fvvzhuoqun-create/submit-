import PyInstaller.__main__

if __name__ == '__main__':
    PyInstaller.__main__.run([
        'gui_app.py',   # 你的主程序文件名
        '--noconsole',  # 运行时不显示控制台窗口
        '--onedir',    # 打包成一个文件夹
        '--hidden-import=tkinter',
        '--hidden-import=tkinter.messagebox'
    ])