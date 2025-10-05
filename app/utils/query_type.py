import enum


class QueryType(enum.Enum):
    # How do I use foo() to read a file?
    USAGE = "USAGE"

    # What does the timeout parameter do in connect()? (API Details)
    PARAMETERS = "PARAMETERS"

    # Why do I get IndexError when calling bar()?
    DEBUGGING = "DEBUGGING"

    # How do I install this libray on macos?
    INSTALLATION = "INSTALLATION"

    # What’s the best way to handle asynchronous tasks with this library?
    BEST_PRACTICES = "BEST_PRACTICES"