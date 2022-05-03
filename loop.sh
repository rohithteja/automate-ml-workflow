for dim_red_type in 'pca' 'lle'; do
    for n_comp in 5 10; do
        for classifier in 'lr' 'svc' 'rf'; do
                echo $dim_red_type $n_comp $classifier 
                python argparse_automate.py --dim_red_type "${dim_red_type}" --n_comp "${n_comp}" --classifier "${classifier}"
        done
    done
done
