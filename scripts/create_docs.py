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
TAG_NAME = "host"
DOC_FILE_EXT = ['rst']
TEMPLATE_FILE_EXT_PREFIX = "t_"
TEMPLATE_FILE_EXT = [TEMPLATE_FILE_EXT_PREFIX + t for t in DOC_FILE_EXT]
DEST_VARIABLE_PREFIX = "filename_"
# Hosts where the docs will be published, e.g. GitHub
# Default filename without extension
HOSTS = {
    'github': {
        'default_filename': "README_github",
        'long_opt_name': "github",
        'hostname': "GitHub",
    },
    'pypi': {
        'default_filename': "README_pypi",
        'long_opt_name': "pypi",
        'hostname': "PyPI",
    },
    'readthedocs': {
        'default_filename': "README_docs",
        'long_opt_name': "docs",
        'hostname': "readthedocs",
    },
}


# Assume template and docs filepaths are valid, already checked
def _create_docs(template_filepath, dst_dirpath, hosts):
    """

    Parameters
    ----------
    template_filepath
    dst_dirpath
    hosts

    Returns
    -------

    """
    orig_txt = read_file(template_filepath)
    regex1 = r"<{} name=\".*?\">.*?<\/{}>".format(TAG_NAME, TAG_NAME)
    # TODO: use threads for each host
    for s in hosts:
        hostname = s[0]
        doc_filename = s[1]
        _, doc_ext = split_fname(doc_filename)
        # Not necessary (since we assume input is cleaned) but just in case
        assert doc_ext, \
            "The doc filename '{}' doesn't have any extension".format(
                doc_filename)
        copy_txt = copy.copy(orig_txt)
        while True:
            # TODO: give better variable names
            match1 = re.search(regex1, copy_txt, re.DOTALL | re.MULTILINE)
            if match1:
                tag = match1.group()
                regex_open_tag = r"<{} name=\".*?\">".format(TAG_NAME)
                match2 = re.search(regex_open_tag, tag,
                                   re.DOTALL | re.MULTILINE)
                if hostname in match2.group():
                    # Case: tags related to selected host
                    close_tag = "</{}>".format(TAG_NAME)
                    first = copy_txt[:match1.start()]
                    middle_start_pos = match1.start()+match2.end()
                    middle_end_pos = match1.end()-len(close_tag)
                    if copy_txt[middle_start_pos] == '\n':
                        # Remove \n, i.e. from open and close tags
                        middle_start_pos += 1
                        middle_end_pos -= 1
                    middle = copy_txt[middle_start_pos:middle_end_pos]
                    last = copy_txt[match1.end():]
                    copy_txt = first + middle + last
                else:
                    # Case: removing tags and their content because not related
                    # to selected host
                    if copy_txt[match1.start()-1] == '\n' and \
                            (len(copy_txt) == match1.end() or  # end of file
                             copy_txt[match1.end()] == '\n'):
                        match1_end = match1.end() + 1
                    else:
                        match1_end = match1.end()
                    copy_txt = copy_txt[:match1.start()] + copy_txt[match1_end:]
            else:
                break
        doc_filepath = os.path.join(dst_dirpath, doc_filename)
        write_file(doc_filepath, copy_txt)
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
    hostnames = ", ".join([s for s in HOSTS.keys()])
    parser.add_argument(
        "-t", "--template", dest="template_filepath",
        required=True,
        help='''File path to the template used for generating the docs from 
        different hosts ({})'''.format(hostnames))
    parser.add_argument(
        "-d", "--dst-dirpath", default=None, dest="dst_dirpath",
        help='''Directory path where the docs ({}) will be saved. By 
        default, they are saved in the directory where the script is 
        executed.'''.format(", ".join(DOC_FILE_EXT)))
    short_opts = []
    for k, v in HOSTS.items():
        hostname = v['hostname']
        doc_filename = v['default_filename']
        if len(DOC_FILE_EXT) == 1:
            default = "{}.{}".format(doc_filename, DOC_FILE_EXT[0])
        else:
            default = "{}. {{{}}}".format(doc_filename, ", ".join(DOC_FILE_EXT))
        dest = "{}{}".format(DEST_VARIABLE_PREFIX, k)
        nargs = "?"
        help_ = '''{}'s doc filename.'''.format(hostname)
        first_letter_name = hostname[0][0].lower()
        parameters = dict(dest=dest,
                          default=default,
                          nargs=nargs,
                          help=help_)
        if first_letter_name in short_opts:
            parser.add_argument("--{}".format(hostname.lower()), **parameters)
        else:
            short_opts.append(hostname[0].lower())
            parser.add_argument(
                "-{}".format(first_letter_name),
                "--{}".format(hostname.lower()),
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
        if args.dst_dirpath is None:
            args.dst_dirpath = ""
        dict_vars = vars(args)
        hosts = []
        _, template_ext = split_fname(args.template_filepath)
        doc_ext = template_ext.split(TEMPLATE_FILE_EXT_PREFIX)[1]
        assert template_ext in TEMPLATE_FILE_EXT, \
            "Wrong template file extension: {}. Valid template file extensions " \
            "are: {}".format(os.path.basename(args.template_filepath),
                             TEMPLATE_FILE_EXT)
        for dest, arg_value in dict_vars.items():
            if dest.startswith(DEST_VARIABLE_PREFIX):
                fname = arg_value
                hostname = dest.lstrip(DEST_VARIABLE_PREFIX)
                if arg_value:
                    fname_root, fname_ext = split_fname(fname)
                    assert fname_ext == doc_ext, \
                        "Template '{}' and doc file '{}' are not " \
                        "compatible. Check their file extensions.".format(
                            os.path.basename(args.template_filepath),
                            fname)
                else:
                    fname = HOSTS[hostname]['default_filename']
                    assert "." not in fname, \
                        "The default filename '{}' should not have an " \
                        "extension".format(fname)
                    fname += ".{}".format(doc_ext)
                if arg_value is None or fname in sys.argv:
                    hosts.append((hostname, fname))
        # =======
        # Actions
        # =======
        if hosts:
            retcode = _create_docs(args.template_filepath, args.dst_dirpath,
                                   hosts)
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
