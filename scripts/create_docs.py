#!/usr/bin/env python
import argparse
import copy
import os
import re
import sys
import traceback

import ipdb

from pyutils.genutils import read_file, write_file


VERSION = "0.0.1a0"
TAG_NAME = "site"
README_FILE_EXT = ['rst']
TEMPLATE_FILE_EXT_PREFIX = "t_"
TEMPLATE_FILE_EXT = [TEMPLATE_FILE_EXT_PREFIX + t for t in README_FILE_EXT]
DEST_VARIABLE_PREFIX = "filename_"
# Websites where the READMEs will be published, e.g. GitHub
# Default filename without extension
WEBSITES = {
    'github': {
        'default_filename': "README_github",
        'long_opt_name': "github",
        'site_name': "GitHub",
    },
    'pypi': {
        'default_filename': "README_pypi",
        'long_opt_name': "pypi",
        'site_name': "PyPI",
    },
    'readthedocs': {
        'default_filename': "README_docs",
        'long_opt_name': "docs",
        'site_name': "readthedocs",
    },
}


# Assume template filepath and README filepaths are valid, already checked
def _create_readme(template_filepath, dst_dirpath, websites):
    """

    Parameters
    ----------
    template_filepath
    dst_dirpath
    websites

    Returns
    -------

    """
    orig_txt = read_file(template_filepath)
    regex1 = r"<{}=\".*?\">.*?<\/{}>".format(TAG_NAME, TAG_NAME)
    # TODO: use threads for each site
    # TODO: add include <site> tags
    for s in websites:
        website_key = s[0]
        readme_filename = s[1]
        _, readme_ext = split_fname(readme_filename)
        # Not necessary (since we assume input is cleaned) but just in case
        assert readme_ext, \
            "The README filename '{}' doesn't have any extension".format(
                readme_filename)
        copy_txt = copy.copy(orig_txt)
        while True:
            # TODO: give better variable names
            match1 = re.search(regex1, copy_txt, re.DOTALL | re.MULTILINE)
            if match1:
                site_tag = match1.group()
                regex_open_site_tag = r"<{}=\".*?\">".format(TAG_NAME)
                match2 = re.search(regex_open_site_tag, site_tag,
                                   re.DOTALL | re.MULTILINE)
                if website_key in match2.group():
                    close_site_tag = "</{}>".format(TAG_NAME)
                    first = copy_txt[:match1.start()]
                    middle_start_pos = match1.start()+match2.end()
                    middle_end_pos = match1.end()-len(close_site_tag)
                    if copy_txt[middle_start_pos] == '\n':
                        # Remove \n, i.e. from open and close tags
                        middle_start_pos += 1
                        middle_end_pos -= 1
                    middle = copy_txt[middle_start_pos:middle_end_pos]
                    last = copy_txt[match1.end():]
                    copy_txt = first + middle + last
                else:
                    # import ipdb
                    # ipdb.set_trace()
                    if copy_txt[match1.start()-1] == '\n' and \
                            (len(copy_txt) == match1.end() or  # end of file
                             copy_txt[match1.end()] == '\n'):
                        # TODO: do a regex search
                        match1_end = match1.end() + 1
                    else:
                        match1_end = match1.end()
                    copy_txt = copy_txt[:match1.start()] + copy_txt[match1_end:]
            else:
                break
        write_file(readme_filename, copy_txt)
    return 0


def setup_argparser():
    """Setup the argument parser for the command-line script.

    The script allows you to.

    Returns
    -------
    args : argparse.Namespace
        Simple class used by default by ``parse_args()`` to create an object
        holding attributes and return it [1]_.

    References
    ----------
    .. [1] `argparse.Namespace
       <https://docs.python.org/3.7/library/argparse.html#argparse.Namespace>`_.

    """
    # Setup the parser
    parser = argparse.ArgumentParser(
        description='''\
.''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # ===============
    # General options
    # ===============
    # TODO: add overwrite option
    parser.add_argument("-v", "--version", action='version',
                        version='%(prog)s {}'.format(VERSION))
    site_names = ", ".join([s for s in WEBSITES.keys()])
    parser.add_argument(
        "-t", "--template", dest="template_filepath",
        required=True,
        help='''File path to the template used for generating the READMEs from
        different sites ({})'''.format(site_names))
    parser.add_argument(
        "-d", "--dst-dirpath", default=None, dest="dst_dirpath",
        help='''Directory path where the READMEs ({}) will be saved. By 
        default, they are saved in the directory where the script is 
        executed.'''.format(", ".join(README_FILE_EXT)))
    short_opts = []
    for k, v in WEBSITES.items():
        site_name = v['site_name']
        site_filename = v['default_filename']
        if len(README_FILE_EXT) == 1:
            default = "{}.{}".format(site_filename, README_FILE_EXT[0])
        else:
            default = "README_{}"
        dest = "{}{}".format(DEST_VARIABLE_PREFIX, k)
        nargs = "?"
        help_ = '''{}'s README filename.'''.format(site_name)
        first_letter_name = site_name[0][0].lower()
        parameters = dict(dest=dest,
                          default=default,
                          nargs=nargs,
                          help=help_)
        if first_letter_name in short_opts:
            parser.add_argument("--{}".format(site_name.lower()), **parameters)
        else:
            short_opts.append(site_name[0].lower())
            parser.add_argument(
                "-{}".format(first_letter_name),
                "--{}".format(site_name.lower()),
                **parameters)
    return parser.parse_args()


def split_fname(fname):
    """TODO

    Parameters
    ----------
    fname

    Returns
    -------

    """
    # TODO: explain code
    root, ext = os.path.splitext(fname)
    ext = ext[1:]
    return root, ext


def main():
    """Main entry-point to the script.

    According to the user's choice of action, the script might

    """
    retcode = 0
    try:
        args = setup_argparser()
        dict_vars = vars(args)
        websites = []
        _, template_ext = split_fname(args.template_filepath)
        readme_ext = template_ext.split(TEMPLATE_FILE_EXT_PREFIX)[1]
        assert template_ext in TEMPLATE_FILE_EXT, \
            "Wrong template file extension: {}. Valid template file extensions " \
            "are: {}".format(os.path.basename(args.template_filepath),
                             TEMPLATE_FILE_EXT)
        for dest, arg_value in dict_vars.items():
            if dest.startswith(DEST_VARIABLE_PREFIX):
                fname = arg_value
                key = dest.lstrip(DEST_VARIABLE_PREFIX)
                if arg_value:
                    fname_root, fname_ext = split_fname(fname)
                    assert fname_ext == readme_ext, \
                        "Template '{}' and README file '{}' are not " \
                        "compatible. Check their file extensions.".format(
                            os.path.basename(args.template_filepath),
                            fname)
                else:
                    fname = WEBSITES[key]['default_filename']
                    assert "." not in fname, \
                        "The default filename '{}' should not have an " \
                        "extension".format(fname)
                    fname_root = fname
                    fname += ".{}".format(readme_ext)
                if arg_value is None or fname in sys.argv:
                    websites.append((key, fname))
        # =======
        # Actions
        # =======
        if websites:
            retcode = _create_readme(args.template_filepath, args.dst_dirpath,
                                     websites)
    except SystemExit:
        retcode = 1
    except Exception:
        retcode = 1
        traceback.print_exc()
    except KeyboardInterrupt:
        # ctrl + c
        pass
    finally:
        return retcode


if __name__ == '__main__':
    retcode = main()
    print("\nProgram exited with {}".format(retcode))
