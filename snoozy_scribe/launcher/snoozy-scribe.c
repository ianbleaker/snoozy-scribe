/*
 * snoozy-scribe.app's executable: runs dictate.py on the repo's .venv inside
 * this process, with Python embedded.
 *
 * A shell launcher that execs .venv/bin/python doesn't work. Homebrew's
 * framework Python re-execs into its own Python.app, so macOS credits
 * Microphone and Accessibility to "python3.14" instead of
 * snoozy-scribe (plan #50, item 10). Embedding keeps the bundle's executable
 * as the process that asks for them.
 *
 * Dev build: make-app.sh passes REPO_DIR (where the .venv and dictate.py
 * are) and LOG_PATH_SUFFIX as defines.
 *
 * Release build (-DRELEASE_BUILD, packaging/build-release-app.sh): no path is
 * baked in. The bundle is found from the executable's own path, Python is
 * the bundled python-build-standalone in Contents/Resources/python, and the
 * process runs Contents/Resources/app/snoozy_scribe/bootstrap.py, which
 * installs the libraries into Application Support on first launch and then
 * runs dictate.py. The app can be moved anywhere.
 *
 * The process runs in $HOME: nothing depends on the cwd, and settings and
 * data live in Application Support.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifdef RELEASE_BUILD
#include <mach-o/dyld.h>
#endif

static void fail(PyConfig *config, PyStatus status) {
    PyConfig_Clear(config);
    Py_ExitStatusException(status);
}

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;

    /* Apps launched from Finder get a bare PATH (/usr/bin:/bin:...), which
       would hide Homebrew's ffmpeg from transcribe.py (the release build
       still uses it for .ogg input). */
    const char *path = getenv("PATH");
    char new_path[4096];
    snprintf(new_path, sizeof new_path, "/opt/homebrew/bin:/usr/local/bin:%s", path ? path : "/usr/bin:/bin");
    setenv("PATH", new_path, 1);

    /* No terminal to show a startup crash, so output goes to a log file. */
    const char *home = getenv("HOME");
    if (home) {
        char log_path[4096];
        snprintf(log_path, sizeof log_path, "%s%s", home, LOG_PATH_SUFFIX);
        freopen(log_path, "a", stdout);
        freopen(log_path, "a", stderr);
        setvbuf(stdout, NULL, _IOLBF, 0);
    }

    if (home && chdir(home) != 0) {
        perror("chdir $HOME");
        return 1;
    }

    PyConfig config;
    PyConfig_InitPythonConfig(&config);
#ifdef RELEASE_BUILD
    /* <bundle>/Contents/MacOS/<exe> -> <bundle>/Contents. realpath resolves
       symlinks, so a Dock alias or a symlinked install finds the real bundle. */
    char exe[PATH_MAX], contents[PATH_MAX];
    uint32_t exe_size = sizeof exe;
    if (_NSGetExecutablePath(exe, &exe_size) != 0 || !realpath(exe, contents)) {
        fprintf(stderr, "can't resolve the executable's path\n");
        return 1;
    }
    for (int i = 0; i < 2; i++) {
        char *slash = strrchr(contents, '/');
        if (!slash) {
            fprintf(stderr, "unexpected executable path: %s\n", contents);
            return 1;
        }
        *slash = '\0';
    }

    char home_dir[PATH_MAX], python[PATH_MAX], stdlib_dir[PATH_MAX], dynload[PATH_MAX];
    char app_dir[PATH_MAX], bootstrap_dir[PATH_MAX], bootstrap_py[PATH_MAX];
    snprintf(home_dir, sizeof home_dir, "%s/Resources/python", contents);
    snprintf(python, sizeof python, "%s/bin/python3", home_dir);
    snprintf(stdlib_dir, sizeof stdlib_dir, "%s/lib/python3.14", home_dir);
    snprintf(dynload, sizeof dynload, "%s/lib/python3.14/lib-dynload", home_dir);
    snprintf(app_dir, sizeof app_dir, "%s/Resources/app", contents);
    snprintf(bootstrap_dir, sizeof bootstrap_dir, "%s/Resources/bootstrap", contents);
    snprintf(bootstrap_py, sizeof bootstrap_py, "%s/snoozy_scribe/bootstrap.py", app_dir);

    PyStatus status = PyConfig_SetBytesString(&config, &config.home, home_dir);
    if (PyStatus_Exception(status)) fail(&config, status);
    status = PyConfig_SetBytesString(&config, &config.executable, python);
    if (PyStatus_Exception(status)) fail(&config, status);

    /* An explicit search path: stdlib (lib-dynload holds _ssl, _ctypes and
       the other compiled modules), then the app code, then the bootstrap's
       PyObjC. bootstrap.py adds the Application Support runtime folder. */
    const char *search[] = {stdlib_dir, dynload, app_dir, bootstrap_dir};
    for (size_t i = 0; i < sizeof search / sizeof *search; i++) {
        wchar_t *w = Py_DecodeLocale(search[i], NULL);
        if (!w) {
            fprintf(stderr, "can't decode path: %s\n", search[i]);
            return 1;
        }
        status = PyWideStringList_Append(&config.module_search_paths, w);
        PyMem_RawFree(w);
        if (PyStatus_Exception(status)) fail(&config, status);
    }
    config.module_search_paths_set = 1;
    /* No user site-packages: ~/.local must never leak into the app. And no
       script folder on sys.path, or app/snoozy_scribe's modules (paths,
       config, ...) would import as top-level modules. */
    config.user_site_directory = 0;
    config.safe_path = 1;

    char *py_argv[] = {python, bootstrap_py};
    status = PyConfig_SetBytesArgv(&config, 2, py_argv);
    if (PyStatus_Exception(status)) fail(&config, status);
    status = PyConfig_SetBytesString(&config, &config.run_filename, bootstrap_py);
    if (PyStatus_Exception(status)) fail(&config, status);
#else
    /* Python's getpath finds .venv/pyvenv.cfg next to the executable, so
       sys.prefix and site-packages come from the venv. */
    PyStatus status = PyConfig_SetBytesString(&config, &config.executable, REPO_DIR "/.venv/bin/python");
    if (PyStatus_Exception(status)) fail(&config, status);
    char *py_argv[] = {REPO_DIR "/.venv/bin/python", REPO_DIR "/dictate.py"};
    status = PyConfig_SetBytesArgv(&config, 2, py_argv);
    if (PyStatus_Exception(status)) fail(&config, status);
    status = PyConfig_SetBytesString(&config, &config.run_filename, REPO_DIR "/dictate.py");
    if (PyStatus_Exception(status)) fail(&config, status);
#endif

    status = Py_InitializeFromConfig(&config);
    if (PyStatus_Exception(status)) fail(&config, status);
    PyConfig_Clear(&config);
    return Py_RunMain();
}
