fn main() {
    let dst = cmake::Config::new("treelite")
        .define("Treelite_BUILD_STATIC_LIBS", "ON")
        .define("Treelite_USE_DYNAMIC_MSVC_RUNTIME", "ON")
        .define("Treelite_INSTALL", "OFF")
        .build();
    println!("cargo:rustc-link-search={}/build", dst.display());
    println!("cargo:rustc-link-lib=static=treelite_static");
    println!("cargo:rustc-link-lib=gomp");
    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-lib=stdc++");
    }
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=c++");
    }
}
