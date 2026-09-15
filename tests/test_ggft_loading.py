from GGFT.ggft import DEFAULT_GFT_TABLE, CubeGFT


def test_default_gft_loads_independently_of_working_directory(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    table = CubeGFT.load_default()

    assert DEFAULT_GFT_TABLE == "cube_tables_n201.npz"
    assert table.resolution == 201
    assert table.probabilities.size == 6 * 201 * 201

