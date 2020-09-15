import pandas as pd
from CCIT import DataGen


def _generate_data(dx, dy, dz, mode="NI"):
    gen_data = DataGen.generate_samples_cos(size=1000, dx=dx, dy=dy, dz=dz, sType=mode)

    y_hat = gen_data[:, :dx]
    y_true = gen_data[:, dx + dy : dx + dy + dz]

    z_protected = gen_data[:, dx : dx + dy]

    _data = {
        f"protected_{i+1}": z_protected[:, i : i + 1].reshape(-1) for i in range(dy)
    }

    _data["y_true"] = y_true.reshape(-1)
    _data["y_hat"] = y_hat.reshape(-1)

    return pd.DataFrame(_data)


def generate_not_independent_data(dx=1, dy=5, dz=1):
    return _generate_data(dx, dy, dz, "NI")


def generate_conditional_independent_data(dx=1, dy=5, dz=1):
    return _generate_data(dx, dy, dz, "CI")


def generate_independent_data(dx=1, dy=5, dz=1):
    return _generate_data(dx, dy, dz, "I")
