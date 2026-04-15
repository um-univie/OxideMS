#[macro_export]
macro_rules! impl_fluent_builder_vanilla {

        (
        impl $struct_name:ident {
            $( $field:ident : $ftype:ty ),* $(,)?
        }
    ) => {
        impl $struct_name {
            $(
                // The method name is exactly the field name
                pub fn $field(mut self, value: $ftype) -> Self {
                    self.$field = value;
                    self
                }
            )*
        }
    };
}

#[macro_export]
macro_rules! struct_with_new {
    (
        // Match optional attributes like #[derive(Debug)]
        $(#[$meta:meta])*
        // Match optional visibility like `pub`
        $struct_vis:vis struct $name:ident {
            $(
                // Match each field: optional visibility, name, and type
                $field_vis:vis $field_name:ident : $field_type:ty
            ),* $(,)? // Allow an optional trailing comma
        }
    ) => {
        // 1. Output the exact struct definition as it was passed in
        $(#[$meta])*
        $struct_vis struct $name {
            $(
                $field_vis $field_name : $field_type,
            )*
        }

        // 2. Generate the `impl` block with the `new` function
        impl $name {
            pub fn new($($field_name : $field_type),*) -> Box<Self> {
                Box::new(Self {
                    $($field_name),*
                })
            }
        }
    };
}
