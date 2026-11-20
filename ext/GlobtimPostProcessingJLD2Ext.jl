"""
    GlobtimPostProcessingJLD2Ext

Reads binary subdivision trees (`Globtim.save_tree`) with `load_subdivision_tree`,
loaded by `using JLD2`. The JSON reader `load_subdivision_result` needs no extension.
"""
module GlobtimPostProcessingJLD2Ext

using GlobtimPostProcessing
import GlobtimPostProcessing: load_subdivision_tree
using JLD2: jldopen

function load_subdivision_tree(filename::AbstractString)
    !isfile(filename) && error("File not found: $filename")

    jldopen(filename, "r") do file
        tree = file["subdivision_tree"]
        metadata = haskey(file, "metadata") ? file["metadata"] : Dict()
        return (tree, metadata)
    end
end

end # module GlobtimPostProcessingJLD2Ext
