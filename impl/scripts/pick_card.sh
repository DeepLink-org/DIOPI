VENDOR=$1

available_cards=()
function get_available_cards_on_supa()
{
    local total_mem=$(brsmi gpu query -d memory | grep Total | awk '{print $3}' | head -n 1)
    local mem_used=$(brsmi gpu query -d memory | grep Used | awk '{print $3}')
    get_available_cards "$total_mem" "$mem_used"
}

function get_available_cards_on_ascend()
{
    local total_mem=$(npu-smi info | grep -E "/.*/" | awk -F '/' '{print $3}' | grep -o [0-9]* | head -n 1 )
    local mem_used=$(npu-smi info | grep -E "/.*/" | awk  '{print $10}' | grep -o '[0-9]*')
    get_available_cards "$total_mem" "$mem_used"
}

function get_available_cards()
{
    local total_mem=$1
    local mem_used=$2
    local mem_used_arr=($mem_used)
    for ((i=0; i<${#mem_used_arr[@]}; i++)); do
        if (( mem_used_arr[i] < $((total_mem/3)) )); then
            available_cards+=($i)
        fi
    done
}

function run()
{
    local vendor=$1
    local env_var_need_export=$2

    get_available_cards_on_"${vendor}"
    # no available card, not export the valiable_devices env var
    if [ "$available_cards" = "" ];then
        echo "no available card to pick, run on the default card"
        return 0
    fi
    echo "available cards are" "${available_cards[@]}"
    local visiable_device="${available_cards[-1]}"  #get the last one
    echo "run on the picked card $visiable_device"

    # export the visiable_devices env var
    export "${env_var_need_export}"="$visiable_device"
}

if [ "$VENDOR" = "supa" ];then
    run supa SUPA_VISIBLE_DEVICES
elif [ "$VENDOR" = "ascend" ];then
    run ascend ASCEND_RT_VISIBLE_DEVICES   # in docker, otherwise ASCEND_VISIBLE_DEVICES
else
    echo "Invalid VENDOR parameter: $VENDOR" 1>&2
fi