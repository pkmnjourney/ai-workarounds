### To workaround the flash attention issue in windows i.e. `No Kernel Found` error.
Go to `transformer.py` located in your conda env under `\Lib\site-packages\sam2\modeling\sam`
and change every instance of the following from:

` # OLD_GPU, USE_FLASH_ATTN, MATH_KERNEL_ON = get_sdpa_settings()`

to:

```USE_FLASH_ATTN = False
MATH_KERNEL_ON = True
OLD_GPU = True```

and from:

`with torch.nn.attention.sdpa_kernel(get_sdp_backends(dropout_p)):`

to:

`with torch.backends.cuda.sdp_kernel(get_sdp_backends(dropout_p)):`