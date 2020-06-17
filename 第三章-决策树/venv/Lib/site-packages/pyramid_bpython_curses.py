import os.path

try:
    from bpython._version import __version__ as version
except ImportError:
    version = 'unknown'

__version__ = version
package_dir = os.path.abspath(os.path.dirname(__file__))


def embed(locals_=None, args=['-i', '-q'], banner=None):
    from bpython.cli import main
    return main(args, locals_, banner)

def bpython_curses_shell_runner(env, help):
    return embed(locals_=env, banner=help + '\n')