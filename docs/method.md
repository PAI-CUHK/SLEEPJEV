# Method

## Runtime path

```text
PSG features
  -> SleepStateEncoder
  -> SleepCache(local/coarse/hour/night tokens)
  -> predicted stage and event serving index
  -> SleepQuery(target, constraints, options)
  -> legal candidate filtering
  -> query-conditioned sparse top-k readout
  -> JEVOptionScorer
  -> probabilities over query.options
```

`SleepStateEncoder` uses epoch features, temporal convolutions, and optional coarse/hour Transformer blocks. `SleepQueryEncoder` embeds runtime query text and its normalized window. `JEVOptionScorer` compares the selected state with each runtime option using one shared scorer and masks padded options.

The event path additionally supports a compiled segment-tree posting cache. Each interval stores local top-k predicted event epochs. A time/stage query decomposes into disjoint nodes and retrieves from their postings without reading the full overnight sequence. Gold event postings are retained for audit and evaluation; runtime candidate selection uses model-produced postings to avoid label leakage.

## What is signal-native?

The signal encoder, temporal hierarchy, event index, stage index, epoch windows, and event localization are signal-side components. The query text, option strings, legal constraints, and option-conditioned score are query-side components. The separation lets one overnight state serve different query lists.

## JEV interpretation

The implementation follows the useful interface idea of making the decision set explicit at runtime. It does not claim to be an official JEV implementation. The project should describe this as **JEV-inspired runtime semantic decision modeling** until the authors provide a formal reference and definition.
