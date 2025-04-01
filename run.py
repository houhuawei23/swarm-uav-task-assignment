from main import init_cmd_args

if __name__ == "__main__":
    cmd_args = init_cmd_args()
    print(cmd_args)
parser.add_argument("--show", action="store_true", help="whether to show the result")
