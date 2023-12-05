# https://stackoverflow.com/a/287944

class BGColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# Maximum Category Text Length
__MAX_CAT_LEN = len("SUCCESS")


def header(text):
    print(f"{BGColors.HEADER}>> {text}{BGColors.ENDC}")


def __category_filler(category):
    return f"[ {category :<{__MAX_CAT_LEN}} ]"


def info(text):
    print(f"{BGColors.BOLD}{BGColors.OKCYAN}{__category_filler('INFO')}{BGColors.ENDC} {text}")


def warn(text):
    print(f"{BGColors.BOLD}{BGColors.WARNING}{__category_filler('WARNING')}{BGColors.ENDC} {text}")


def error(text):
    print(f"{BGColors.BOLD}{BGColors.FAIL}{__category_filler('ERROR')}{BGColors.ENDC} {text}")


def ok(text):
    print(f"{BGColors.BOLD}{BGColors.OKBLUE}{__category_filler('OK')}{BGColors.ENDC} {text}")


def success(text):
    print(f"{BGColors.BOLD}{BGColors.OKGREEN}{__category_filler('SUCCESS')}{BGColors.ENDC} {text}")


if __name__ == "__main__":
    # Testing
    # print(f"{BGColors.WARNING}Warning: No active frommets remain. Continue?{BGColors.ENDC}")
    header("Welcome to the test program!")
    info("Something is happening...")
    warn("Something is not right...")
    error("Something bad happened!")
    ok("Something is okay!")
    success("Something was a success!")
