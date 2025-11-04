import React from 'react'
import styled from 'styled-components'
import { theme } from '../theme'

const FooterContainer = styled.footer`
  background: linear-gradient(135deg, ${theme.colors.purple}, ${theme.colors.teal});
  padding: ${theme.spacing.xxl} ${theme.spacing.lg};
  margin-top: ${theme.spacing.xxl};
  border-top: 3px solid ${theme.colors.mint};
`

const FooterContent = styled.div`
  max-width: 1600px;
  margin: 0 auto;
`

const FooterGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: ${theme.spacing.xl};
  margin-bottom: ${theme.spacing.xl};
  
  @media (max-width: ${theme.breakpoints.mobile}) {
    grid-template-columns: 1fr;
  }
`

const FooterSection = styled.div`
  color: ${theme.colors.light};
`

const FooterTitle = styled.h3`
  color: ${theme.colors.light};
  font-size: 1.3rem;
  margin-bottom: ${theme.spacing.md};
  border-bottom: 2px solid ${theme.colors.light};
  padding-bottom: ${theme.spacing.sm};
`

const FooterText = styled.p`
  line-height: 1.8;
  margin-bottom: ${theme.spacing.sm};
  font-size: 0.95rem;
`

const FooterLink = styled.a`
  color: ${theme.colors.mint};
  text-decoration: none;
  transition: color ${theme.transitions.fast};
  
  &:hover {
    color: ${theme.colors.light};
  }
`

const TechStack = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: ${theme.spacing.sm};
  margin-top: ${theme.spacing.md};
`

const TechBadge = styled.span`
  background: rgba(23, 34, 38, 0.7);
  padding: ${theme.spacing.sm} ${theme.spacing.md};
  border-radius: ${theme.borderRadius.sm};
  font-size: 0.85rem;
  font-weight: bold;
`

const Copyright = styled.div`
  text-align: center;
  padding-top: ${theme.spacing.xl};
  border-top: 1px solid rgba(178, 228, 217, 0.3);
  color: ${theme.colors.light};
  font-size: 0.9rem;
`

const Footer = () => {
  return (
    <FooterContainer>
      <FooterContent>
        <FooterGrid>
          <FooterSection>
            <FooterTitle>Author</FooterTitle>
            <FooterText><strong>Cazandra Aporbo, MS</strong></FooterText>
            <FooterText>Head of Data Science, FoXX Health</FooterText>
            <FooterText>
              <strong>Education:</strong><br />
              MS in Data Science, University of Denver<br />
              BS in Integrative Biology, Oregon State University
            </FooterText>
            <FooterText>
              <strong>Specialization:</strong><br />
              AI Ethics in Healthcare<br />
              Bias Detection in Medical Systems<br />
              Predictive Modeling for Women's Health
            </FooterText>
          </FooterSection>

          <FooterSection>
            <FooterTitle>Project Details</FooterTitle>
            <FooterText><strong>Dataset:</strong> Nobel Prize Winners (1901-2025)</FooterText>
            <FooterText><strong>Records:</strong> 995 awards across 125 years</FooterText>
            <FooterText><strong>Analysis Date:</strong> November 3, 2025</FooterText>
            <FooterText><strong>Version:</strong> 1.0</FooterText>
            <FooterText>
              <FooterLink href="https://github.com/Cazzy-Aporbo/Nobel-Peace-Prize" target="_blank" rel="noopener noreferrer">
                View on GitHub
              </FooterLink>
            </FooterText>
          </FooterSection>

          <FooterSection>
            <FooterTitle>Technical Stack</FooterTitle>
            <TechStack>
              <TechBadge>React 18</TechBadge>
              <TechBadge>Recharts 2.10</TechBadge>
              <TechBadge>Styled Components</TechBadge>
              <TechBadge>Vite 5</TechBadge>
              <TechBadge>Python 3.11</TechBadge>
              <TechBadge>Pandas</TechBadge>
              <TechBadge>Scikit-learn</TechBadge>
            </TechStack>
            <FooterText style={{ marginTop: theme.spacing.md }}>
              <strong>Analytical Dimensions:</strong><br />
              Temporal Evolution, Gender Disparity, Geospatial Clustering, 
              Collaboration Networks, NLP, Machine Learning, Dimensionality Reduction, 
              Statistical Inference
            </FooterText>
          </FooterSection>
        </FooterGrid>

        <Copyright>
          © 2025 Cazandra Aporbo. All rights reserved. | MIT License
          <br />
          <small>
            This analysis demonstrates comprehensive expertise in temporal analysis, bias detection, 
            geospatial methods, collaboration networks, natural language processing, machine learning, 
            dimensionality reduction, and statistical inference applied to complex real-world datasets.
          </small>
        </Copyright>
      </FooterContent>
    </FooterContainer>
  )
}

export default Footer
