#!/bin/bash
# By yskoh@mellanox.com

# IFG_PLUS_PREAMBLE=20 # IFG 12B + Preamble 8B
IFG_PLUS_PREAMBLE=26 # IFG 12B + Preamble 8B

update_stats () { # $name $index
	TS_LAST[$2]=${TS[$2]}
	R_PKT_LAST[$2]=${R_PKT[$2]}
	R_BYTE_LAST[$2]=${R_BYTE[$2]}
	T_PKT_LAST[$2]=${T_PKT[$2]}
	T_BYTE_LAST[$2]=${T_BYTE[$2]}

	# ETHTOOL=($(ethtool -S $1 | awk '/tx_packets_phy/{print $2} /rx_packets_phy/{print $2} /tx_bytes_phy/{print $2} /rx_bytes_phy/{print$2}'))
	# if [ -z "$ETHTOOL" ]; then
	# 	ETHTOOL=($(ethtool -S $1 | awk '/tx_packets/{print $2} /rx_packets/{print $2} /tx_bytes/{print $2} /rx_bytes/{print$2}'))
	# fi
    

	TS[$2]=$(date +%s%6N) # in usec
	T_PKT[$2]=$(cat /sys/class/infiniband/mlx5_1/ports/1/counters/port_xmit_packets)
	R_PKT[$2]=$(cat /sys/class/infiniband/mlx5_1/ports/1/counters/port_rcv_packets)
	T_BYTE[$2]=$(cat /sys/class/infiniband/mlx5_1/ports/1/counters/port_xmit_data)
	R_BYTE[$2]=$(cat /sys/class/infiniband/mlx5_1/ports/1/counters/port_rcv_data)
}

# if [ -z $1 ] && [ -x /usr/bin/ibdev2netdev ]; then
# 	NETIF=$(ibdev2netdev | awk '/mlx/{print $5}')
# else
# 	NETIF=$@
# fi

NETIF=mlx5_1

if [ -z "$NETIF" ]; then
	printf "No interface can't be found\n"
	exit 1
fi

# set initial value
index=0
for name in $NETIF; do 
	update_stats $name $index
	((index++))
done

index=0
for name in $NETIF; do 
	R_PKT_INIT[$index]=${R_PKT[$index]}
	T_PKT_INIT[$index]=${T_PKT[$index]}
	R_BYTE_INIT[$index]=${R_BYTE[$index]}
	T_BYTE_INIT[$index]=${T_BYTE[$index]}
	((index++))
done

sleep 1

while true; do

	index=0
	for name in $NETIF; do 
		update_stats $name $index

		TS_DIFF=$((${TS[$index]} - ${TS_LAST[$index]}))

		R_PKT_DELTA=$(( ${R_PKT[$index]} - ${R_PKT_LAST[$index]} ))
		R_PKT_RATE=$(( $R_PKT_DELTA * 1000000 / $TS_DIFF))

		R_BIT_DELTA=$(( (${R_BYTE[$index]} - ${R_BYTE_LAST[$index]} + $IFG_PLUS_PREAMBLE * $R_PKT_DELTA) * 8 * 4 ))
		R_BIT_RATE=$(( $R_BIT_DELTA * 1000000 / $TS_DIFF))

		T_PKT_DELTA=$(( ${T_PKT[$index]} - ${T_PKT_LAST[$index]} ))
		T_PKT_RATE=$(( $T_PKT_DELTA * 1000000 / $TS_DIFF))

		T_BIT_DELTA=$(( (${T_BYTE[$index]} - ${T_BYTE_LAST[$index]} + $IFG_PLUS_PREAMBLE * $T_PKT_DELTA) * 8 * 4 ))
		T_BIT_RATE=$(( $T_BIT_DELTA * 1000000 / $TS_DIFF))

		R_PKT_TOTAL=$(( ${R_PKT[$index]} - ${R_PKT_INIT[$index]} ))
		T_PKT_TOTAL=$(( ${T_PKT[$index]} - ${T_PKT_INIT[$index]} ))

		R_BYTE_TOTAL=$(( ${R_BYTE[$index]} - ${R_BYTE_INIT[$index]} ))
		T_BYTE_TOTAL=$(( ${T_BYTE[$index]} - ${T_BYTE_INIT[$index]} ))

		printf "[%'9s Rx]: %'16d pkts  %'16d pps  | %'20d bytes %'16d bps \n" $name $R_PKT_TOTAL $R_PKT_RATE $R_BYTE_TOTAL $R_BIT_RATE
		printf "[%'9s Tx]: %'16d pkts  %'16d pps  | %'20d bytes %'16d bps \n" $name $T_PKT_TOTAL $T_PKT_RATE $T_BYTE_TOTAL $T_BIT_RATE

		((index++))
	done

	printf "\n"
	sleep 1
done
