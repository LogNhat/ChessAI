/^\[Event/ {
    if (game != "" && (white >= 2700 || black >= 2700)) {
        print game "\n"
    }
    game = ""
    white = 0
    black = 0
}

{
    game = game $0 "\n"

    if ($0 ~ /\[WhiteElo "/) {
        match($0, /[0-9]+/)
        white = substr($0, RSTART, RLENGTH)
    }

    if ($0 ~ /\[BlackElo "/) {
        match($0, /[0-9]+/)
        black = substr($0, RSTART, RLENGTH)
    }
}

END {
    if (game != "" && (white >= 2700 || black >= 2700)) {
        print game
    }
}