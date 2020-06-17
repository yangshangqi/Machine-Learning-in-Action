#!F:\代码文件集\python代码文件\机器学习实战\第三章-决策树\venv\Scripts\python.exe
# EASY-INSTALL-ENTRY-SCRIPT: 'curses-2048==1.4','console_scripts','2048'
__requires__ = 'curses-2048==1.4'
import re
import sys
from pkg_resources import load_entry_point

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(
        load_entry_point('curses-2048==1.4', 'console_scripts', '2048')()
    )
