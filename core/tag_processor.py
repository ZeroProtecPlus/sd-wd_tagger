import re
from typing import List, Tuple, Dict, Set
from core.config import WDTaggerConfig

class TagProcessor:
    """Handles tag processing and formatting"""
    
    def __init__(self, config: WDTaggerConfig):
        self.config = config
        self.r34_tag_mappings = self._init_r34_mappings()
        self.nsfw_indicators = self._init_nsfw_indicators()
        self.quality_tags = self._init_quality_tags()
        self.style_tags = self._init_style_tags()
    
    def _init_r34_mappings(self) -> Dict[str, str]:
        """Initialize R34-specific tag mappings"""
        return {
            # Body parts and anatomy
            "large breasts": "large_breasts",
            "huge breasts": "huge_breasts",
            "small breasts": "small_breasts",
            "medium breasts": "medium_breasts",
            "flat chest": "flat_chest",
            "wide hips": "wide_hips",
            "thick thighs": "thick_thighs",
            "long legs": "long_legs",
            "big ass": "big_ass",
            "curvy": "curvy",
            
            # Clothing and accessories
            "school uniform": "school_uniform",
            "sailor uniform": "sailor_uniform",
            "gym uniform": "gym_uniform",
            "maid outfit": "maid_outfit",
            "bikini": "bikini",
            "one piece swimsuit": "one-piece_swimsuit",
            "lingerie": "lingerie",
            "stockings": "stockings",
            "thigh highs": "thigh_highs",
            "knee highs": "knee_highs",
            "pantyhose": "pantyhose",
            "high heels": "high_heels",
            "boots": "boots",
            "gloves": "gloves",
            "hair bow": "hair_bow",
            "hair ribbon": "hair_ribbon",
            "glasses": "glasses",
            "cat ears": "cat_ears",
            "animal ears": "animal_ears",
            
            # Poses and expressions
            "looking at viewer": "looking_at_viewer",
            "looking away": "looking_away",
            "looking back": "looking_back",
            "spread legs": "spread_legs",
            "arms up": "arms_up",
            "arms behind back": "arms_behind_back",
            "hand on hip": "hand_on_hip",
            "peace sign": "peace_sign",
            "wink": "wink",
            "blush": "blush",
            "smile": "smile",
            "open mouth": "open_mouth",
            "tongue out": "tongue_out",
            
            # Hair and physical features
            "long hair": "long_hair",
            "short hair": "short_hair",
            "twin tails": "twin_tails",
            "ponytail": "ponytail",
            "hair bun": "hair_bun",
            "ahoge": "ahoge",
            "bangs": "bangs",
            "side bangs": "side_bangs",
            "hair over one eye": "hair_over_one_eye",
            "red hair": "red_hair",
            "blonde hair": "blonde_hair",
            "brown hair": "brown_hair",
            "black hair": "black_hair",
            "blue hair": "blue_hair",
            "green hair": "green_hair",
            "pink hair": "pink_hair",
            "purple hair": "purple_hair",
            "white hair": "white_hair",
            "silver hair": "silver_hair",
            "red eyes": "red_eyes",
            "blue eyes": "blue_eyes",
            "green eyes": "green_eyes",
            "brown eyes": "brown_eyes",
            "yellow eyes": "yellow_eyes",
            "purple eyes": "purple_eyes",
            "pink eyes": "pink_eyes",
            
            # Settings and backgrounds
            "school": "school",
            "classroom": "classroom",
            "bedroom": "bedroom",
            "bathroom": "bathroom",
            "kitchen": "kitchen",
            "outdoors": "outdoors",
            "beach": "beach",
            "pool": "pool",
            "park": "park",
            "city": "city",
            "night": "night",
            "day": "day",
            "sunset": "sunset",
            "sunrise": "sunrise",
            
            # Art style and quality
            "anime": "anime",
            "manga": "manga",
            "realistic": "realistic",
            "photorealistic": "photorealistic",
            "detailed": "detailed",
            "high quality": "high_quality",
            "masterpiece": "masterpiece",
            "best quality": "best_quality",
            "official art": "official_art",
            "illustration": "illustration",
            "artwork": "artwork",
            "digital art": "digital_art",
            "traditional art": "traditional_art"
        }
    
    def _init_nsfw_indicators(self) -> Set[str]:
        """Initialize NSFW indicator tags"""
        return {
            "nude", "naked", "topless", "bottomless", "underwear", "lingerie",
            "bikini", "swimsuit", "cleavage", "nipples", "areolae", "pussy",
            "penis", "ass", "buttocks", "thighs", "panties", "bra", "sex",
            "cum", "orgasm", "masturbation", "vibrator", "dildo", "bondage",
            "bdsm", "tentacles", "rape", "forced", "gangbang", "orgy",
            "futanari", "shemale", "transgender", "yaoi", "yuri", "hentai",
            "ecchi", "lewd", "nsfw", "explicit", "adult", "mature"
        }
    
    def _init_quality_tags(self) -> Set[str]:
        """Initialize quality indicator tags"""
        return {
            "masterpiece", "best quality", "high quality", "detailed",
            "ultra detailed", "extremely detailed", "highly detailed",
            "amazing", "beautiful", "gorgeous", "stunning", "perfect",
            "flawless", "incredible", "outstanding", "exceptional",
            "professional", "official art", "promotional art"
        }
    
    def _init_style_tags(self) -> Set[str]:
        """Initialize style indicator tags"""
        return {
            "anime", "manga", "realistic", "photorealistic", "semi-realistic",
            "cartoon", "chibi", "sketch", "line art", "cel shading",
            "soft shading", "hard shading", "watercolor", "oil painting",
            "digital art", "traditional art", "concept art", "illustration",
            "cg", "3d", "2d", "pixiv", "danbooru", "gelbooru"
        }
    
    def clean_tag(self, tag: str) -> str:
        """Clean and normalize a single tag"""
        # Remove extra spaces and normalize
        tag = re.sub(r'\s+', ' ', tag.strip())
        
        # Handle parentheses (escape for some formats)
        tag = tag.replace('(', '\\(').replace(')', '\\)')
        
        return tag
    
    def format_standard_tags(self, tag_results: List[Tuple[str, float]]) -> str:
        """Format tags in standard comma-separated format"""
        if not tag_results:
            return ""
        
        # Sort by confidence
        sorted_tags = sorted(tag_results, key=lambda x: x[1], reverse=True)
        
        # Clean and format tags
        formatted_tags = []
        for tag, confidence in sorted_tags:
            cleaned_tag = self.clean_tag(tag)
            formatted_tags.append(cleaned_tag)
        
        return ", ".join(formatted_tags)
    
    def format_r34_tags(self, tag_results: List[Tuple[str, float]]) -> str:
        """Format tags specifically for R34 and similar platforms"""
        if not tag_results:
            return ""
        
        # Sort by confidence
        sorted_tags = sorted(tag_results, key=lambda x: x[1], reverse=True)
        
        # Process tags for R34 format
        r34_tags = []
        for tag, confidence in sorted_tags:
            # Apply R34-specific mappings
            if tag in self.r34_tag_mappings:
                r34_tag = self.r34_tag_mappings[tag]
            else:
                # Convert spaces to underscores for R34 format
                r34_tag = tag.replace(' ', '_')
                
                # Remove special characters that might cause issues
                r34_tag = re.sub(r'[^\w\-_]', '', r34_tag)
                
                # Handle multiple underscores
                r34_tag = re.sub(r'_+', '_', r34_tag)
                
                # Remove leading/trailing underscores
                r34_tag = r34_tag.strip('_')
            
            if r34_tag and r34_tag not in r34_tags:
                r34_tags.append(r34_tag)
        
        return " ".join(r34_tags)
    
    def categorize_tags(self, tag_results: List[Tuple[str, float]]) -> Dict[str, List[str]]:
        """Categorize tags by type"""
        categories = {
            "character": [],
            "clothing": [],
            "pose": [],
            "background": [],
            "style": [],
            "quality": [],
            "nsfw": [],
            "other": []
        }
        
        clothing_keywords = {"uniform", "outfit", "clothes", "dress", "shirt", "skirt", "pants", "jacket", "coat", "hat", "shoes", "boots", "socks", "stockings", "gloves", "accessories"}
        pose_keywords = {"sitting", "standing", "lying", "kneeling", "running", "walking", "dancing", "jumping", "flying", "pose", "position"}
        background_keywords = {"room", "school", "house", "building", "street", "park", "beach", "mountain", "forest", "sky", "cloud", "water", "background"}
        
        for tag, confidence in tag_results:
            tag_lower = tag.lower()
            
            # Check categories
            if any(keyword in tag_lower for keyword in clothing_keywords):
                categories["clothing"].append(tag)
            elif any(keyword in tag_lower for keyword in pose_keywords):
                categories["pose"].append(tag)
            elif any(keyword in tag_lower for keyword in background_keywords):
                categories["background"].append(tag)
            elif tag_lower in self.quality_tags:
                categories["quality"].append(tag)
            elif tag_lower in self.style_tags:
                categories["style"].append(tag)
            elif tag_lower in self.nsfw_indicators:
                categories["nsfw"].append(tag)
            else:
                categories["other"].append(tag)
        
        return categories
    
    def filter_tags_by_confidence(self, tag_results: List[Tuple[str, float]], min_confidence: float = 0.1) -> List[Tuple[str, float]]:
        """Filter tags by minimum confidence threshold"""
        return [(tag, conf) for tag, conf in tag_results if conf >= min_confidence]
    
    def get_top_tags(self, tag_results: List[Tuple[str, float]], n: int = 20) -> List[Tuple[str, float]]:
        """Get top N tags by confidence"""
        sorted_tags = sorted(tag_results, key=lambda x: x[1], reverse=True)
        return sorted_tags[:n]
    
    def remove_duplicate_tags(self, tag_results: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Remove duplicate tags, keeping the one with higher confidence"""
        seen_tags = {}
        
        for tag, confidence in tag_results:
            normalized_tag = tag.lower().strip()
            if normalized_tag not in seen_tags or seen_tags[normalized_tag][1] < confidence:
                seen_tags[normalized_tag] = (tag, confidence)
        
        return list(seen_tags.values())
    
    def enhance_r34_tags(self, tag_results: List[Tuple[str, float]]) -> str:
        """Enhanced R34 tag formatting with better categorization"""
        if not tag_results:
            return ""
        
        # Remove duplicates and filter by confidence
        filtered_tags = self.remove_duplicate_tags(tag_results)
        filtered_tags = self.filter_tags_by_confidence(filtered_tags, 0.1)
        
        # Categorize tags
        categorized = self.categorize_tags(filtered_tags)
        
        # Build R34 tag string with priority order
        r34_parts = []
        
        # High priority tags first
        priority_order = ["character", "quality", "style", "clothing", "pose", "background", "other"]
        
        for category in priority_order:
            if categorized[category]:
                category_tags = []
                for tag in categorized[category]:
                    r34_tag = self.r34_tag_mappings.get(tag, tag.replace(' ', '_'))
                    r34_tag = re.sub(r'[^\w\-_]', '', r34_tag)
                    r34_tag = re.sub(r'_+', '_', r34_tag).strip('_')
                    
                    if r34_tag and r34_tag not in category_tags:
                        category_tags.append(r34_tag)
                
                r34_parts.extend(category_tags)
        
        # Add NSFW tags at the end if present
        if categorized["nsfw"]:
            nsfw_tags = []
            for tag in categorized["nsfw"]:
                r34_tag = tag.replace(' ', '_')
                r34_tag = re.sub(r'[^\w\-_]', '', r34_tag)
                r34_tag = re.sub(r'_+', '_', r34_tag).strip('_')
                
                if r34_tag and r34_tag not in nsfw_tags:
                    nsfw_tags.append(r34_tag)
            
            r34_parts.extend(nsfw_tags)
        
        return " ".join(r34_parts)