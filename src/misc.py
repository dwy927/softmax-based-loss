import os
from contextlib import contextmanager
from pathlib import Path

def ensure_dir(*paths, erase=False):
    import shutil
    for path in paths:
        if os.path.exists(path) and erase:
            print('Removing old folder {}'.format(path))
            try:
                shutil.rmtree(path)
            except Exception as e:
                print('Try to use sudo')
                import traceback
                traceback.print_exc()
                os.system('sudo rm -rf {}'.format(path))

        if not os.path.exists(path):
            print('Creating folder {}'.format(path))
            try:
                os.makedirs(path, exist_ok=True)
            except Exception as e:
                print('Try to use sudo')
                import traceback
                traceback.print_exc()
                os.system('sudo mkdir -p {}'.format(path))

@contextmanager
def change_dir(dirpath):
    dirpath = str(dirpath)
    cwd = os.getcwd()
    try:
        os.chdir(dirpath)
        yield
    finally:
        os.chdir(cwd)


def make_symlink_if_not_exists(src, dst, overwrite=False):
    src, dst = Path(src), Path(dst)

    if overwrite and dst.is_symlink():
        dst.unlink()

    if (not dst.exists()) and (not dst.is_symlink()):
        dst.symlink_to(src)
    else:
        raise OSError("symbolic link {} already exists!".format(dst))