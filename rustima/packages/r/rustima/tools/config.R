Sys.setenv(PATH = paste(Sys.getenv("PATH"), path.expand("~/.cargo/bin"), sep = .Platform$path.sep))
for (tool in c("cargo", "rustc")) {
    if (!nzchar(Sys.which(tool))) stop("Install ", tool, " before installing rustima")
    version <- system2(tool, "--version", stdout = TRUE)
    message(version)
    if (tool == "rustc") {
        actual <- strsplit(version, " ", fixed = TRUE)[[1]][2]
        if (utils::compareVersion(actual, "1.87") < 0L) stop("rustima requires rustc >= 1.87")
    }
}
if (!file.exists("src/rust/vendor.tar.xz"))
    stop("Use tools/build_r_package.py to prepare the self-contained R source package")
utils::untar("src/rust/vendor.tar.xz", exdir = "src")
dir.create("src/.cargo", showWarnings = FALSE)
file.copy("src/rust/vendor-config.toml", "src/.cargo/config.toml", overwrite = TRUE)
windows <- .Platform$OS.type == "windows"
template <- if (windows) "src/Makevars.win.in" else "src/Makevars.in"
target <- sub("[.]in$", "", template)
file.copy(template, target, overwrite = TRUE)
