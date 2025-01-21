from .infer import predict_cli

class CmdEntries:

    @property
    def predict(self):
        return predict_cli
