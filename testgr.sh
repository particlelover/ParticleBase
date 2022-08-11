#! /bin/sh

EXEP=../TamaF/

echo -n 'benchmark start '
date

# dummy
${EXEP}/testMD1 > /dev/null 2> /dev/null

if [ ! -e MD_CO2done ];
then
  ${EXEP}/calcMD_CO2 > /dev/null 2> /dev/null
fi

${EXEP}/calcgr1 MD_CO2done > /dev/null 2> log.gr1_bench
tail -n 3 log.gr1_bench
