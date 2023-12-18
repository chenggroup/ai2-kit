import pydantic


class BaseModel(pydantic.BaseModel):
    class Config:
        extra = pydantic.Extra.forbid