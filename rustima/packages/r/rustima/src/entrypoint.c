/* Keep Rust's routine registration reachable from R's loader. */
void R_init_rustima_extendr(void *dll);
void register_extendr_panic_hook(void);

void R_init_rustima(void *dll) {
    register_extendr_panic_hook();
    R_init_rustima_extendr(dll);
}
