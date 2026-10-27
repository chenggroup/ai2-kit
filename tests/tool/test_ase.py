from ai2_kit.tool.ase import AseTool


def test_write_each_frame_delegates_to_write_frames():
    tool = AseTool.__new__(AseTool)

    assert tool.write_each_frame == tool.write_frames
