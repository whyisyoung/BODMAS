import sys
from datetime import datetime
from fabric2 import Connection

def main():
    if len(sys.argv) != 6:
        print('You need to specify host, script, train_set, classifier, seed, for example: python -u fabric_pretrain.py storm run_pretrain.sh ember gbdt 0')
    else:
        host = sys.argv[1]
        script = sys.argv[2]
        train_set = sys.argv[3]
        classifier = sys.argv[4]
        seed = sys.argv[5]

        conn = Connection(host)

        # WARNING: change the following line to where your script was stored
        with conn.cd('~/BODMAS/code/'):
            result = conn.run(f'./{script} {train_set} {classifier} {seed}')
            msg = "Ran {0.command!r} on {0.connection.host}, got stdout:\n{0.stdout}"
            print(msg.format(result))

if __name__ == "__main__":
    main()
