
class CmdEntries:

    @property
    def viber(self):
        from .viber import cmd_entry
        return cmd_entry


def cli_main():
    import fire
    fire.Fire(CmdEntries)
