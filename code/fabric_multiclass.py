import sys
from datetime import datetime
from fabric2 import Connection

def main():
    if len(sys.argv) != 5:
        print('You need to specify host, script, classifier, families_cnt, for example: python -u fabric.py storm run_multiclass.sh gbdt 5')
    else:
        host = sys.argv[1]
        script = sys.argv[2]
        classifier = sys.argv[3]
        families_cnt = sys.argv[4]

        conn = Connection(host)

        # WARNING: change the following line to where your script was stored
        with conn.cd('~/BODMAS/code/'):
            result = conn.run(f'./{script} {classifier} {families_cnt}')
            msg = "Ran {0.command!r} on {0.connection.host}, got stdout:\n{0.stdout}"
            print(msg.format(result))

if __name__ == "__main__":
    main()
