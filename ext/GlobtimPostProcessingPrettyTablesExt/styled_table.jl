# Table printers moved from src/display_helpers.jl (bead 0do2).
# The documented catch-all stubs stay in src.

function styled_table(
    io::IO,
    data;
    header,
    alignment,
    highlighters::Tuple = (),
    tf = tf_unicode_rounded,
    header_crayon = PrettyTables.Crayon(bold = true),
    crop::Symbol = :none,
    kwargs...,
)
    pretty_table(
        io,
        data;
        header = header,
        alignment = alignment,
        highlighters = highlighters,
        tf = tf,
        header_crayon = header_crayon,
        crop = crop,
        kwargs...,
    )
end

# Convenience: default io=stdout
function styled_table(data; kwargs...)
    styled_table(stdout, data; kwargs...)
end
