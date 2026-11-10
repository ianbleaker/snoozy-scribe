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
 * Built by make-app.sh, which passes REPO_DIR and LOG_PATH_SUFFIX as defines.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static void fail(PyConfig *config, PyStatus status) {
    PyConfig_Clear(config);
    Py_ExitStatusException(status);
}

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;

    /* Apps launched from Finder get a bare PATH (/usr/bin:/bin:...), which
       would hide Homebrew's ffmpeg from transcribe.py. */
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

    if (chdir(REPO_DIR) != 0) {
        perror("chdir " REPO_DIR);
        return 1;
    }

    PyConfig config;
    PyConfig_InitPythonConfig(&config);
    /* Python's getpath finds .venv/pyvenv.cfg next to the executable, so
       sys.prefix and site-packages come from the venv. */
    PyStatus status = PyConfig_SetBytesString(&config, &config.executable, REPO_DIR "/.venv/bin/python");
    if (PyStatus_Exception(status)) fail(&config, status);
    char *py_argv[] = {REPO_DIR "/.venv/bin/python", REPO_DIR "/dictate.py"};
    status = PyConfig_SetBytesArgv(&config, 2, py_argv);
    if (PyStatus_Exception(status)) fail(&config, status);
    status = PyConfig_SetBytesString(&config, &config.run_filename, REPO_DIR "/dictate.py");
    if (PyStatus_Exception(status)) fail(&config, status);

    status = Py_InitializeFromConfig(&config);
    if (PyStatus_Exception(status)) fail(&config, status);
    PyConfig_Clear(&config);
    return Py_RunMain();
}
