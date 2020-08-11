import argparse
import traceback

from pyutils.genutils import read_file


VERSION = "0.0.0a0"


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
    parser.add_argument("-v", "--version", action='version',
                        version='%(prog)s {}'.format(VERSION))
    return parser.parse_args()


def main():
    """Main entry-point to the script.

    According to the user's choice of action, the script might

    Notes
    -----
    Only one action at a time can be performed.

    """
    # =======
    # Actions
    # =======
    retcode = 0
    try:
        """
        if args.example_number == 1:
            ex1_turn_on_led(args.led_channel[0], args.time_led_on)
        else:
            print("Example # {} not found".format(args.example_number))
        """
        if True:
            pass
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
