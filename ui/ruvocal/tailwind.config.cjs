const defaultTheme = require("tailwindcss/defaultTheme");
const colors = require("tailwindcss/colors");

/** @type {import('tailwindcss').Config} */
module.exports = {
	darkMode: "class",
	mode: "jit",
	content: ["./src/**/*.{html,js,svelte,ts}"],
	theme: {
		extend: {
			fontFamily: {
				sans: ['Inter', ...defaultTheme.fontFamily.sans],
				mono: ['ui-monospace', 'SF Mono', 'Cascadia Code', 'Fira Code', ...defaultTheme.fontFamily.mono],
			},
			colors: {
				gray: {
					600: "#323843",
					700: "#1a1d24",
					800: "#0f1115",
					900: "#080a0d",
					950: "#020205",
				},
				// RuVector gold/amber accent (matches pi.ruv.io)
				gold: {
					DEFAULT: "#e8a634",
					50: "#fef9ec",
					100: "#fdf0c8",
					200: "#fbe08c",
					300: "#f9cc4f",
					400: "#f0d89a",
					500: "#e8a634",
					600: "#d18a1a",
					700: "#ae6817",
					800: "#8e511a",
					900: "#754319",
					950: "#432209",
				},
			},
			fontSize: {
				xxs: "0.625rem",
				smd: "0.94rem",
			},
			animation: {
				'pulse-gold': 'pulse-glow 4s ease infinite',
				'float': 'float 3s ease-in-out infinite',
				'pixel-in': 'pixelIn 0.6s cubic-bezier(0.16, 1, 0.3, 1) both',
			},
			keyframes: {
				'pulse-glow': {
					'0%, 100%': { opacity: '0.8', filter: 'drop-shadow(0 0 6px #e8a634)' },
					'50%': { opacity: '0.5', filter: 'drop-shadow(0 0 2px #e8a634)' },
				},
				'float': {
					'0%, 100%': { transform: 'translateY(0)' },
					'50%': { transform: 'translateY(-4px)' },
				},
				'pixelIn': {
					'0%': { filter: 'blur(8px)', opacity: '0', transform: 'scale(1.1)' },
					'30%': { filter: 'blur(4px)', opacity: '0.5' },
					'60%': { filter: 'blur(1px)', opacity: '0.8' },
					'100%': { filter: 'blur(0)', opacity: '1', transform: 'scale(1)' },
				},
			},
		},
	},
	plugins: [
		require("tailwind-scrollbar")({ nocompatible: true }),
		require("@tailwindcss/typography"),
	],
};
