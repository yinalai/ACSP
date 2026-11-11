import sys

from attack_symbols import main


if __name__ == "__main__":
    if any(s == "--attack-mode" or s.startswith("--attack-mode=")
           for s in sys.argv[1:]):
        raise SystemExit("This PK entrypoint fixes --attack-mode=pk")
    sys.argv.extend(["--attack-mode", "pk"])
    main()
