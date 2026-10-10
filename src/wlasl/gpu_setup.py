"""Rend les DLL CUDA visibles pour onnxruntime sur Windows.

Les paquets pip nvidia-*-cu12 deposent leurs DLL dans
site-packages/nvidia/<lib>/bin, qui n est pas dans le chemin de recherche.
onnxruntime charge onnxruntime_providers_cuda.dll dynamiquement, et c est
le chargeur de Windows qui resout SES dependances (cublasLt64_12.dll,
cudnn64_9.dll) — or os.add_dll_directory ne couvre pas ce cas, seul PATH
le fait.

Sans cela, onnxruntime n echoue pas : il annonce CUDAExecutionProvider
comme disponible, puis retombe silencieusement sur le CPU. On passe de
25 ms a 780 ms par image sans le moindre message d erreur.

A importer AVANT onnxruntime.
"""

import os
import site


def enable_cuda_dlls():
    """Ajoute les dossiers de DLL CUDA au PATH. Retourne les chemins ajoutes."""
    directories = []

    roots = list(site.getsitepackages())
    user_site = site.getusersitepackages()
    if isinstance(user_site, str):
        roots.append(user_site)

    for root in roots:
        base = os.path.join(root, "nvidia")
        if not os.path.isdir(base):
            continue

        for package in sorted(os.listdir(base)):
            binary_dir = os.path.join(base, package, "bin")
            if os.path.isdir(binary_dir):
                directories.append(binary_dir)
        break

    if directories:
        os.environ["PATH"] = (
            os.pathsep.join(directories) + os.pathsep + os.environ.get("PATH", "")
        )
        for directory in directories:
            try:
                os.add_dll_directory(directory)
            except (OSError, AttributeError):
                pass

    return directories


def check():
    """Verifie que CUDA tourne REELLEMENT, pas qu il est annonce.

    Un provider annonce comme disponible peut quand meme retomber sur le
    CPU a la creation de session : c est le cas qui nous a coute une heure.
    """
    enable_cuda_dlls()

    import numpy as np
    import onnxruntime as ort

    available = ort.get_available_providers()
    if "CUDAExecutionProvider" not in available:
        return False, f"CUDAExecutionProvider absent : {available}"

    # Un provider annonce peut quand meme retomber sur le CPU : on cree
    # une vraie session pour le verifier. On utilise un modele deja present
    # plutot qu un graphe synthetique, pour ne pas dependre du paquet onnx.
    import glob

    models = glob.glob(
        os.path.expanduser("~/.cache/rtmlib/hub/checkpoints/*.onnx")
    )
    if not models:
        return True, "provider annonce (aucun modele local pour le verifier)"

    try:
        session = ort.InferenceSession(
            models[0], providers=["CUDAExecutionProvider"]
        )
        used = session.get_providers()[0]
        return used == "CUDAExecutionProvider", f"session utilise {used}"
    except Exception as error:
        return False, f"{type(error).__name__}: {str(error)[:120]}"


if __name__ == "__main__":
    ok, detail = check()
    print(("CUDA actif : " if ok else "CUDA INACTIF : ") + detail)
