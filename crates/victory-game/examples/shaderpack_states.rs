//! Read-only render-state audit for a shader pack or the archive holding it.

use std::collections::BTreeMap;
use std::error::Error;
use std::path::PathBuf;

use victory_game::cityhash::city64;
use victory_game::pipelines::{self, Named, Unnamed};
use victory_game::{sgd5, shaderpack};

fn main() -> Result<(), Box<dyn Error>> {
    let path = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: shaderpack_states <shader archive>")?;
    let archive = std::fs::read(path)?;
    let bytes = shaderpack::find(&archive).ok_or("shader pack not found")?;
    let pack = shaderpack::Pack::parse(&bytes)?;
    let databases = sgd5::databases(&archive);
    let named = Named::find(&pack, &databases);
    let mut built = 0;
    let mut missing = BTreeMap::<Unnamed, usize>::new();

    for (index, graphics) in pack.graphics().enumerate() {
        match pipelines::stream(&graphics, &named, &[], false, None) {
            Ok(_) => built += 1,
            Err(reason) => {
                *missing.entry(reason).or_default() += 1;
                eprintln!(
                    "graphics {index}: {reason:?}, depth {:016x}, raster {:016x}, blend {:016x}",
                    graphics.depth_stencil_hash, graphics.rasterizer_hash, graphics.blend_hash,
                );
            }
        }
    }

    println!(
        "{} SGD5 databases, {} graphics records, {} compute records",
        databases.len(),
        pack.graphics().count(),
        pack.compute().count(),
    );
    println!(
        "{} depth, {} rasterizer, {} blend states",
        named.depth.len(),
        named.raster.len(),
        named.blend.len(),
    );
    println!("{built} graphics descriptions assembled, {missing:?} unresolved");
    if !named
        .depth
        .iter()
        .map(|(hash, bytes)| (hash, bytes.as_slice()))
        .chain(
            named
                .raster
                .iter()
                .map(|(hash, bytes)| (hash, bytes.as_slice())),
        )
        .chain(
            named
                .blend
                .iter()
                .map(|(hash, bytes)| (hash, bytes.as_slice())),
        )
        .all(|(&hash, bytes)| city64(bytes) == hash)
    {
        return Err("render state hash mismatch".into());
    }
    if !missing.is_empty() {
        return Err("could not assemble every archived graphics description".into());
    }
    Ok(())
}
