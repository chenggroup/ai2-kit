class CmdEntries:

    @property
    def viber(self):
        """
        Viber specific tools.
        """
        from .viber import cmd_entry
        return cmd_entry
