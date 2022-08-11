#! /bin/sh

EXEP=../TamaF/

echo -n 'benchmark start '
date

# dummy
${EXEP}/testMD1 > /dev/null 2> /dev/null

for EXE in SPH DEM MD
do
    for o in 4 4n 5 5D 6 6m
    do
        EXEC=${EXE}${o}
        if [ -x ${EXEP}/test${EXEC} ]
        then
            P=`echo ${EXE} |cut -c 1`${o}
            echo ${EXEC}
            ${EXEP}/test${EXEC} > dump.${P} 2> log.${P}_bench
            if [ $? = 0 ];
            then
                tail -n 3 log.${P}_bench
            else
                echo -n "aborted by: "
                tail -n 1 log.${P}_bench
            fi
            rm -f ${EXEC}done
        fi
    done
done
