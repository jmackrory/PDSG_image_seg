import subprocess

    
def get_img_size(path_str):
    """get tuple for width, height of image"""
    cmd_list=['identify', '-format', '(%w,%h)', "{}".format(path_str)]
    size_str=subprocess.run(cmd_list,stdout=subprocess.PIPE)
    return eval(size_str.stdout.decode('utf-8'))

def make_img_index(index_str,dir_str):

    #make a glob of files.

    #traverse tree with paths.  

    #make list of filename, size, class, and number of subparts.
    return None


def load_img_index(index_str):
    """load Python dict with images including path, size"""
    return None

